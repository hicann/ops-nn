# SwigluGroupQuantWithDualAxis

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

### 接口功能

该算子融合Clipped-SwiGLU、可选逐行weight乘法以及**两个方向**的MX FP8量化，一次调用输出五路结果：

- `y1`、`mxScale1`：沿最后一维（-1轴）量化，每个token行内每32个元素共享一个尺度。
- `y2`、`mxScale2`：沿行方向（-2轴）量化；group场景按`group_index`的分组边界，组内每32行、每列共享一个尺度；non-group场景按全体行成块。
- `yOrigin`：可选的乘weight前激活结果。

第一路的量化数值与单轴算子`SwigluGroupQuant`的`quantMode=5` **逐字节一致**（两者复用同一MX kernel），
因此训练侧（本算子）与推理侧（单轴`quantMode=5`）的第一路结果对齐。

### 计算公式

Clipped-SwiGLU激活可统一写为：

$$
A'[t,h] =
\begin{cases}
\min(A[t,h], L), & L > 0 \\
A[t,h], & L = -1
\end{cases}
$$

$$
B'[t,h] =
\begin{cases}
\min(\max(B[t,h], -L), L), & L > 0 \\
B[t,h], & L = -1
\end{cases}
$$

$$
F[t, h] = A'[t,h] \times \sigma\!\left(\alpha \times A'[t,h]\right)
\times \left(B'[t,h] + \beta\right)
$$

其中：

- $A = x[:, :H]$、$B = x[:, H:]$，$H = D/2$
- $L$ 为`clampLimit`，仅支持`-1.0`（关闭Clamp）或有限正数
- $\alpha$ 为`alpha`，$\beta$ 为`bias`
- $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ 为Sigmoid

**基础计算流程**

```txt
步骤一：输入切分（沿末维前后切分为 A、B）
步骤二：Clamp处理（可选）
步骤三：变体SwiGLU激活
步骤四：Weight加权（可选）
步骤五：双轴MX FP8量化（第一路 -1 轴 / 第二路 -2 轴）
```

<details>
<summary><strong>步骤一：输入切分</strong></summary>

输入张量 $\mathbf{x} \in \mathbb{R}^{T \times D}$ 沿最后一维前后切分为两部分（$H = D/2$）：

$$
A[t, h] = \mathbf{x}[t, h], \quad h \in [0, H)
$$

$$
B[t, h] = \mathbf{x}[t, h + H], \quad h \in [0, H)
$$

</details>

<details>
<summary><strong>步骤二：Clamp处理（可选）</strong></summary>

当`clampLimit > 0`时：

$$
A'[t, h] = \min(A[t, h], L)
$$

$$
B'[t, h] = \min(\max(B[t, h], -L), L)
$$

当`clampLimit = -1.0`时不启用Clamp，此时 $A'=A$、$B'=B$。

**Clamp语义**：门控分支 $A$ 仅做上界截断，即 $A' \in (-\infty, L]$；线性分支 $B$ 做对称截断，即 $B' \in [-L, L]$。

</details>

<details>
<summary><strong>步骤三：变体SwiGLU激活</strong></summary>

$$
F[t, h] = A'[t, h] \cdot \sigma\!\left(\alpha \cdot A'[t, h]\right) \cdot \left(B'[t, h] + \beta\right)
$$

当 $\alpha = 1.0$、$\beta = 0.0$ 且`clampLimit = -1.0`（关闭Clamp）时，计算退化为标准SwiGLU。激活在FP32精度下计算，
最终结果舍入到输入dtype（记为 $R_T(\cdot)$）：

$$
\text{yOrigin} = R_T(F)
$$

`output_origin=true`时输出该值（不含weight影响），可直接用于反向计算`grad_weight`。

</details>

<details>
<summary><strong>步骤四：Weight加权（可选）</strong></summary>

当提供`weight`时：

$$
u[t, h] = R_T\!\left(R_T(F[t, h]) \cdot w[t]\right)
$$

`weight`缺省时 $u = R_T(F)$。$u$ 即两个量化分支共用的量化输入，在`x.dtype`（FP16/BF16）下物化。

</details>

<details>
<summary><strong>步骤五：双轴MX FP8量化</strong></summary>

两路量化均遵循CuBALS（`scaleAlg=1`）尺度算法，块大小固定为32。
量化前先计算块内绝对值最大值，再映射到目标FP8类型的可表示范围：

$$
s^{raw} = \frac{amax}{Amax(DType)}
$$

其中 $Amax(DType)$：FP8_E4M3FN为 $448.0$，FP8_E5M2为 $57344.0$。

对于有限非零的 $s^{raw}$，将其按FP32位模式解释。记 $E_b$ 为FP32的**8 bit偏置指数域**（biased exponent），
$m \in [0,1)$ 为尾数域对应的小数部分。CuBALS对指数执行向上取整：

$$
E_b^{*} =
\begin{cases}
E_b + 1, & 0 < E_b < 254 \text{ 且 } m > 0 \\
E_b + 1, & E_b = 0 \text{ 且 } m > 0.5 \\
E_b, & \text{否则}
\end{cases}
$$

E8M0直接保存该偏置指数编码：

$$
mxScale = E_b^{*}, \qquad s = 2^{E_b^{*}-127}
$$

块内数据按 $d = Q8(u / s)$ 量化，FP32→FP8采用**RINT（就近舍入）**。
若块内 $amax = 0$（全零块），则`mxScale = 0`；未使用的scale padding/预留位置不属于有效结果，调用方不应依赖其值。

<details>
<summary><strong>第一路（-1轴，`y1` / `mxScale1`）</strong></summary>

每个token行的行内，沿末维每32个元素成块，各行独立计算尺度：

$$
amax[t, c] = \max_{i=0}^{31} |u[t, 32c + i]|
$$

$$
mxScale1[t, c] = \text{CuBALS}(amax[t, c]), \qquad y1[t, j] = Q8\!\left(\frac{u[t, j]}{s[t, \lfloor j/32 \rfloor]}\right)
$$

`mxScale1`物理排布为`[T, ceil(ceil(H/32)/2), 2]`，最后一维最多存放同一行相邻的2个32-element block scale。若最后一个pair未填满，未使用位置无有效值保证，调用方不应读取或依赖其内容。

</details>

<details>
<summary><strong>第二路（-2轴，`y2` / `mxScale2`）</strong></summary>

**group场景**：按`group_index`给出的分组边界，组内每32行成块，同一列共享一个尺度，
块不跨组；组末不足32行时仅在尺度计算的临时块内按0补齐，不产生额外的`y2`有效行：

$$
amax[g, b, j] = \max_{0 \le i < \min(32,\ e_g-e_{g-1}-32b)} |u[e_{g-1} + 32b + i,\ j]|
$$

$$
y2[t, j] = Q8\!\left(\frac{u[t, j]}{s\big[g(t),\ \lfloor (t - e_{g(t)-1})/32 \rfloor,\ j\big]}\right)
$$

**non-group场景**：退化为全体行按32行成块，每列独立求尺度。

`mxScale2`的物理shape为：group场景`[floor(T/64) + G, H, 2]`，non-group场景`[ceil(T/64), H, 2]`。
最后一维的`slot0/slot1`最多对应两个相邻的32-row block scale。group场景为保持分组边界会保留额外的物理位置；
空组、奇数个32-row block的尾部以及其他未使用的预留位置均无有效值保证，调用方不应读取或依赖其内容。

</details>

</details>

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 130px">
  <col style="width: 130px">
  <col style="width: 430px">
  <col style="width: 190px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>SWIGLU输入，最后一维前后均分为A、B两部分。shape为[T, 2H]，非空，2H大于等于64且能被64整除。支持非连续Tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入（可选）</td>
      <td>逐token权重，非空时乘到激活结果上。仅group场景支持，此时必须同时提供group_index。维度为1-8维，元素个数需等于T。支持非连续Tensor。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>输入（可选）</td>
      <td>累积组端索引（cumsum语义，元素值本身即组端行索引），仅用于第二路（-2轴）分组量化，不承担输出截断。shape为1维[G]，非空，元素非负、非递减、末元素等于T，相邻相等表示空组；不传表示non-group。支持非连续Tensor。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>属性</td>
      <td>目标量化类型。ACLNN/GE编码支持35、36，分别表示FLOAT8_E5M2、FLOAT8_E4M3FN；Torch编码支持291（float8_e5m2）、292（float8_e4m3fn），默认292，实现同时兼容35、36。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>属性</td>
      <td>量化模式。仅支持1，表示双轴MX FP8量化。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clamp_limit</td>
      <td>属性</td>
      <td>Clipped-SwiGLU的clamp阈值。默认-1.0表示不启用Clamp；启用时必须为有限正数。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_origin</td>
      <td>属性</td>
      <td>是否输出乘weight前的激活yOrigin。默认false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>属性</td>
      <td>变体SwiGLU的sigmoid输入缩放系数。默认1.0，必须为有限正数。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>属性</td>
      <td>变体SwiGLU的线性分支偏置。默认0.0，必须为有限数。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>第一路（-1轴）量化输出。shape为[T, H]。</td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxScale1</td>
      <td>输出</td>
      <td>第一路量化尺度。shape为[T, ceil(ceil(H/32)/2), 2]。最后一维最多存放同一行相邻的2个32元素block scale；未使用的padding位置无有效值保证，调用方不应依赖其内容。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>第二路（-2轴）量化输出。shape为[T, H]。</td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxScale2</td>
      <td>输出</td>
      <td>第二路量化尺度。group场景shape为[floor(T/64) + G, H, 2]，non-group场景为[ceil(T/64), H, 2]，G为group_index的元素个数。最后一维slot0/slot1最多对应两个相邻的32行block scale；group场景可能包含为分组边界保留的物理位置，未使用的预留槽位无有效值保证，调用方不应依赖其内容。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>yOrigin</td>
      <td>输出</td>
      <td>乘weight前的激活结果。output_origin为true时shape为[T, H]，数据类型与x一致；false时为[0]占位。不支持空指针。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `x`仅支持二维`[T, 2H]`、FLOAT16或BFLOAT16，非空，`2H >= 64`且能被64整除。
- `quant_mode`仅支持`1`；`dst_type`仅支持FLOAT8_E4M3FN（默认）与FLOAT8_E5M2。
- `weight`仅在提供`group_index`时允许，元素数必须等于`T`；non-group场景传`weight`将被拒绝。
- `group_index`为累积组端索引，元素值在**设备侧**执行时校验：非法值（负值、递减、越界、末元素不等于`T`）
  会触发设备执行异常，异步调用可能在后续流同步时才报告失败；失败后所有输出均不可使用。
  `GetWorkspaceSize`阶段只校验dtype、维数与元素个数，不保证元素值合法。
- `clamp_limit = -1.0`关闭Clamp，否则必须为有限正数；`alpha`必须为有限正数，`bias`必须有限。
- 仅前向，不支持autograd。
- 默认支持确定性计算。
- `x`、`weight`、`group_index`均支持非连续Tensor（内部自动转连续）。
- 输出中`H = D/2 = x.shape[-1]/2`，`G = group_index.numel()`；`mxScale1`、`mxScale2`中未使用的padding或预留位置无有效值保证，调用方不应依赖其内容。

## 调用说明

|调用方式|调用样例|说明|
|:-------|:-------|:---|
|aclnn调用|[test_aclnn_swiglu_group_quant_with_dual_axis](./examples/test_aclnn_swiglu_group_quant_with_dual_axis.cpp)|通过[aclnnSwigluGroupQuantWithDualAxis](./docs/aclnnSwigluGroupQuantWithDualAxis.md)接口调用SwigluGroupQuantWithDualAxis算子。|
|图模式调用|-|通过[算子IR](./op_graph/swiglu_group_quant_with_dual_axis_proto.h)构图方式调用SwigluGroupQuantWithDualAxis算子。|
