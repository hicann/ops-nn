# aclnnSwigluGroupQuantV2

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/activation/swiglu_group_quant)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

### 接口功能

`aclnnSwigluGroupQuantV2`是 [aclnnSwigluGroupQuant](aclnnSwigluGroupQuant.md) 的V2接口，
对应**`quantMode=5`（mx_quant_v2）**这一独立量化模式，在V1的MX量化基础上新增：

- 新增`alpha`、`bias`两个属性，支持变体ClippedSwiGLU激活
  $A' \cdot \sigma(\alpha \cdot A') \cdot (B' + \beta)$；
- `weightOptional`的数据类型由FLOAT32扩展为**FLOAT16、BFLOAT16、FLOAT32**；
- 量化尺度采用**CuBALS**（`scale_alg=1`，无amax下限），与双轴算子
  `swiglu_group_quant_with_dual_axis`的第1路共享同一MX kernel，
  因此训练（双轴）与推理（单轴`quantMode=5`）的第1路量化数值保持一致。

> **与V1的关系**：`quantMode=5`是独立模式，不由V1的参数组合推导。
> V1的`quantMode=0/1/2/3`走`aclnnSwigluGroupQuant`；只有`quantMode=5`走本接口。
> `alpha=1.0`、`bias=0.0`且weight为FLOAT32（或不传）时，本接口的激活与
> V1的`quantMode=1`一致，但量化尺度编码不同（见“量化计算”）。

### 计算公式

**基础计算流程**

```txt
步骤一：输入切分
步骤二：Clamp处理（可选）
步骤三：变体SwiGLU激活
步骤四：Weight加权（可选）
步骤五：MX FP8量化（CuBALS）
```

> `quantMode=5`不支持`group_index`与`scale`，因此无分组截断与静态量化步骤；
> 输入`x`必须为二维，全部`T`行参与计算。

<details>
<summary><strong>步骤一：输入切分</strong></summary>

输入张量 $\mathbf{x} \in \mathbb{R}^{T \times D}$ 沿最后一维切分为两部分（$H = D/2$）：

$$
\mathbf{x}_0[t, h] = \mathbf{x}[t, h], \quad h \in [0, H)
$$

$$
\mathbf{x}_1[t, h] = \mathbf{x}[t, h + H], \quad h \in [0, H)
$$

</details>

<details>
<summary><strong>步骤二：Clamp处理（可选）</strong></summary>

当`clampLimit > 0`时，对输入进行限制：

$$
\mathbf{x}_0'[t, h] = \min(\mathbf{x}_0[t, h], c)
$$

$$
\mathbf{x}_1'[t, h] = \min(\max(\mathbf{x}_1[t, h], -c), c)
$$

其中 $c$ 为`clampLimit`。`clampLimit = -1.0`时不启用。

**Clamp的作用**：

- $\mathbf{x}_0$（门控分支）限制为正值范围 $[0, c]$，防止sigmoid梯度消失
- $\mathbf{x}_1$（线性分支）限制为对称范围 $[-c, c]$，防止数值溢出

</details>

<details>
<summary><strong>步骤三：变体SwiGLU激活</strong></summary>

激活函数（逐元素计算），带`alpha`、`bias`的变体形式：

$$
\mathbf{y}_{\text{swiglu}}[t, h] = \mathbf{x}_0'[t, h] \cdot \sigma\!\left(\alpha \cdot \mathbf{x}_0'[t, h]\right) \cdot \left(\mathbf{x}_1'[t, h] + \beta\right)
$$

其中：

- $\alpha$ 为`alpha`
- $\beta$ 为`bias`
- $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ 为Sigmoid函数

当 $\alpha = 1.0$、$\beta = 0.0$ 时退化为标准SwiGLU：

$$
\mathbf{y}_{\text{swiglu}}[t, h] = \text{Swish}(\mathbf{x}_0'[t, h]) \cdot \mathbf{x}_1'[t, h],
\quad \text{Swish}(z) = z \cdot \sigma(z)
$$

**完整计算步骤分解**：

$$
\begin{aligned}
t_1[t, h] &= -\alpha \cdot \mathbf{x}_0'[t, h] \quad \text{(muls)} \\
t_2[t, h] &= e^{t_1[t, h]} = e^{-\alpha \cdot \mathbf{x}_0'[t, h]} \quad \text{(exp)} \\
t_3[t, h] &= t_2[t, h] + 1 = 1 + e^{-\alpha \cdot \mathbf{x}_0'[t, h]} \quad \text{(adds)} \\
t_4[t, h] &= \frac{\mathbf{x}_0'[t, h]}{t_3[t, h]} = \mathbf{x}_0'[t, h] \cdot \sigma\!\left(\alpha \cdot \mathbf{x}_0'[t, h]\right) \quad \text{(div)} \\
t_5[t, h] &= \mathbf{x}_1'[t, h] + \beta \quad \text{(adds)} \\
\mathbf{y}_{\text{swiglu}}[t, h] &= t_4[t, h] \cdot t_5[t, h] \quad \text{(mul)}
\end{aligned}
$$

**yOrigin输出**：

`outputOrigin`设置为True时，`yOrigin`输出SwiGLU的结果 $\mathbf{y}_{\text{swiglu}}$
（**weight加权前**，但已包含clamp、alpha、bias的作用），可直接用于反向计算`grad_weight`。

</details>

<details>
<summary><strong>步骤四：Weight加权（可选）</strong></summary>

当提供`weight`时，对SwiGLU输出进行加权：

$$
\mathbf{y}_{\text{weighted}}[t, h] = \mathbf{y}_{\text{swiglu}}[t, h] \cdot w[t]
$$

其中 $w[t]$ 为第 $t$ 个token的weight值，可按FLOAT16、BFLOAT16、FLOAT32传入，
参与计算时按FP32精度处理。

**MoE场景**：weight来自专家路由器的softmax输出，表示该token对当前专家的权重。

</details>

<details>
<summary><strong>步骤五：MX FP8量化（CuBALS）</strong></summary>

**MX量化原理**：采用**E8M0 Scale** + **FP8 Data**的组合。

**分组方式**：每**32元素**为一组，不足32的元素补0参与计算：

$$
\mathbf{y} = [\mathbf{g}_0, \mathbf{g}_1, \ldots, \mathbf{g}_K], \quad \mathbf{g}_i \in \mathbb{R}^{32}
$$

**Amax计算**（CuBALS无下限）：

$$
a_i = \max_{j=0}^{31} |\mathbf{g}_i[j]|
$$

**原始Scale计算**：

$$
s_i^{\text{raw}} = \frac{a_i}{M_{\text{fp8}}}
$$

其中 $M_{\text{fp8}}$ 取值：

- FP8 E4M3FN：$M_{\text{fp8}} = 448.0$
- FP8 E5M2：$M_{\text{fp8}} = 57344.0$

**指数向上取整（CuBALS）**：从 $s_i^{\text{raw}}$ 提取无偏指数 $E_i$ 与尾数 $M_i$，
为保证量化不溢出对指数向上取整：

$$
E_i^{*} =
\begin{cases}
E_i + 1, & s_i^{\text{raw}} \text{为正规数，且 } E_i < 254 \text{ 且 } M_i > 0 \\
E_i + 1, & s_i^{\text{raw}} \text{为非正规数，且 } M_i > 0.5 \\
E_i, & \text{否则}
\end{cases}
$$

$$
s_i = 2^{E_i^{*}}
$$

**E8M0 Scale编码**：$s_i$ 以无偏指数形式存入FLOAT8_E8M0：

$$
s_i^{\text{e8m0}} = E_i^{*} + 127
$$

**全零块处理**：$a_i = 0$ 时 $s_i^{\text{e8m0}} = 0$（无amax下限，与双轴算子一致）。

**量化计算**：

$$
\mathbf{y}_{\text{quant}}[t, j] = \text{cast\_fp8\_rint}\!\left(\mathbf{y}_{\text{weighted}}[t, j] \cdot \frac{1}{s_i}\right),
\quad j \in \text{group } i
$$

其中`cast_fp8_rint`为FP32到FP8的类型转换，采用**RINT（就近舍入）**模式。

</details>

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnSwigluGroupQuantV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSwigluGroupQuantV2”接口执行计算。

```Cpp
aclnnStatus aclnnSwigluGroupQuantV2GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weightOptional,
  const aclTensor *groupIndexOptional,
  const aclTensor *scaleOptional,
  int64_t          dstType,
  int64_t          quantMode,
  int64_t          blockSize,
  bool             roundScale,
  double           clampLimit,
  double           dstTypeMax,
  bool             outputOrigin,
  double           alpha,
  double           bias,
  const aclTensor *yOut,
  const aclTensor *yScaleOut,
  const aclTensor *yOriginOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSwigluGroupQuantV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnSwigluGroupQuantV2GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
  <col style="width: 260px">
  <col style="width: 110px">
  <col style="width: 220px">
  <col style="width: 420px">
  <col style="width: 180px">
  <col style="width: 90px">
  <col style="width: 100px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>SwiGLU输入。</td>
      <td><ul><li><b>必须为二维</b>，shape为[T, D]。</li><li>D须<b>大于等于64且能被64整除</b>。</li><li>不支持空Tensor（T须大于0）。</li><li>数据类型为FLOAT16、BFLOAT16。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>MOE权重张量，用于SwiGLU输出的加权计算。</td>
      <td><ul><li>可选参数。</li><li>不为空时，数据类型为<b>FLOAT16、BFLOAT16或FLOAT32</b>，元素个数需等于T。</li><li><b>FLOAT16、BFLOAT16为V2新增能力</b>。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>分组索引。</td>
      <td><ul><li><b>quantMode=5不支持，必须传nullptr。</b></li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scaleOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>静态量化invScale。</td>
      <td><ul><li><b>quantMode=5不支持，必须传nullptr。</b></li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dstType（int64_t）</td>
      <td>输入</td>
      <td>目标量化类型。</td>
      <td><ul><li>支持取值35、36，分别表示FLOAT8_E5M2、FLOAT8_E4M3FN。</li><li>不支持FP4类型（FLOAT4_E2M1/FLOAT4_E1M2）。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td><ul><li>支持取值0、1、2、3、5。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize（int64_t）</td>
      <td>输入</td>
      <td>量化块大小。</td>
      <td><ul><li>0表示使用当前量化模式的默认block大小。</li><li>quantMode为5时，支持0或32。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundScale（bool）</td>
      <td>输入</td>
      <td>是否将scale取整为2的幂。</td>
      <td><ul><li>quantMode为5时，roundScale必须为true。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clampLimit（double）</td>
      <td>输入</td>
      <td>SwiGLU计算前的clamp阈值。</td>
      <td><ul><li>-1.0表示不启用clamp。</li><li>启用clamp时，clampLimit必须大于0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax（double）</td>
      <td>输入</td>
      <td>目标量化类型的最大有限值。</td>
      <td><ul><li>quantMode为5时该参数不生效。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputOrigin（bool）</td>
      <td>输入</td>
      <td>是否输出weight加权前的SwiGLU结果。</td>
      <td><ul><li>true表示输出yOrigin，为false时yOrigin输出无效。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha（double）</td>
      <td>输入</td>
      <td><b>V2新增。</b>SwiGLU激活中Sigmoid输入的缩放系数。</td>
      <td><ul><li>需为有限正数（alpha &gt; 0）。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias（double）</td>
      <td>输入</td>
      <td><b>V2新增。</b>SwiGLU激活中线性分支的偏置。</td>
      <td><ul><li>需为有限数。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut（aclTensor*）</td>
      <td>输出</td>
      <td>量化输出。</td>
      <td><ul><li>shape为[T, D/2]。</li><li>数据类型需与dstType一致。</li></ul></td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>yScaleOut（aclTensor*）</td>
      <td>输出</td>
      <td>量化scale输出。</td>
      <td><ul><li>shape为[T, ceil(ceil((D/2)/32)/2), 2]，最后一维每2个scale成对存放。</li><li>数据类型为FLOAT8_E8M0。</li><li>未使用的padding位置填0。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>yOriginOut（aclTensor*）</td>
      <td>输出</td>
      <td>weight加权前的SwiGLU结果。</td>
      <td><ul><li>shape为[T, D/2]。</li><li>数据类型需与x一致。</li><li>不支持空指针。</li><li>outputOrigin为false时不写入内容，但shape仍为[T, D/2]。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x、yOut、yScaleOut、yOriginOut、workspaceSize或executor存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>输入或输出的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>输入或输出的shape不满足约束（例如x非二维、D未按64对齐）。</td>
    </tr>
    <tr>
      <td>dstType、quantMode、blockSize、roundScale或clampLimit不符合当前支持的值。</td>
    </tr>
    <tr>
      <td>alpha不是有限正数，或bias不是有限数。</td>
    </tr>
    <tr>
      <td>weightOptional不满足可选输入约束，或非法传入了groupIndexOptional/scaleOptional。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>多个输入tensor之间的shape信息不匹配、输入属性不在取值范围（详见参数说明）。</td>
    </tr>
  </tbody></table>

## aclnnSwigluGroupQuantV2

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSwigluGroupQuantV2GetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：aclnnSwigluGroupQuantV2默认确定性实现。
- **量化模式**：支持`quantMode=0/1/2/3/5`；`dstType`支持35（FLOAT8_E5M2）与36（FLOAT8_E4M3FN）。
- **输入形状**：`x`必须为二维`[T, D]`，`T > 0`，`D >= 64`且`D % 64 == 0`。
- **blockSize**支持0或32；**roundScale**必须为true。
- **不支持** `groupIndexOptional`（分组截断）与`scaleOptional`（静态量化），二者必须传`nullptr`。
- **weightOptional**支持FLOAT16、BFLOAT16、FLOAT32，元素数为`T`。
- **`yOriginOut`语义**：输出的是weight加权前的SwiGLU激活结果（含clamp、alpha、bias的作用），
  与`weight`取值无关，可直接用于反向计算`grad_weight`。
- **尺度算法**：固定CuBALS（`scale_alg=1`），无amax下限；全零块的scale编码为0。
  该算法与双轴算子`swiglu_group_quant_with_dual_axis`的第1路复用同一kernel，
  保证训练与推理的第1路量化数值一致。
- **与V1的差异**：`quantMode=5`只在本接口支持；`aclnnSwigluGroupQuant`的
  `quantMode=1`保留了原有的scale编码（amax下限 $10^{-4}$），二者在同一输入下的
  量化scale可能不同（全零块及amax极小的块）。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_swiglu_group_quant_v2.h"

namespace SwigluExample {
inline int64_t Numel(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        if (dim <= 0 || size > std::numeric_limits<int64_t>::max() / dim) {
            throw std::invalid_argument("tensor shape must be positive and fit int64");
        }
        size *= dim;
    }
    return size;
}

inline bool CheckHardwareSupport(const char* operatorName)
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        std::cout << "Warning: cannot get SOC name, skip hardware check" << std::endl;
        return true;
    }
    std::cout << "Current SOC: " << socName << std::endl;
    if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
        return true;
    }
    std::cout << "Warning: " << operatorName << " only supports Ascend950, current SOC '" << socName
              << "' is not supported. Skip test." << std::endl;
    return false;
}

class Session {
public:
    explicit Session(int& status) : status_(status) {}
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session()
    {
        if (pending_) {
            Record(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
        }
        if (executor_ != nullptr) {
            Record(aclDestroyAclOpExecutor(executor_), "aclDestroyAclOpExecutor");
        }
        if (workspace_ != nullptr) {
            Record(aclrtFree(workspace_), "aclrtFree(workspace)");
        }
        for (auto& tensor : tensors_) {
            if (tensor->descriptor != nullptr) {
                Record(aclDestroyTensor(tensor->descriptor), "aclDestroyTensor");
            }
            if (tensor->device != nullptr) {
                Record(aclrtFree(tensor->device), "aclrtFree(tensor)");
            }
        }
        if (stream_ != nullptr) {
            Record(aclrtDestroyStream(stream_), "aclrtDestroyStream");
        }
        if (deviceSet_) {
            Record(aclrtResetDevice(deviceId_), "aclrtResetDevice");
        }
        if (initialized_) {
            Record(aclFinalize(), "aclFinalize");
        }
    }

    void Check(int result, const char* operation)
    {
        if (result != ACL_SUCCESS) {
            Record(result, operation);
            throw std::runtime_error(operation);
        }
    }

    void Init(int32_t deviceId)
    {
        deviceId_ = deviceId;
        Check(aclInit(nullptr), "aclInit");
        initialized_ = true;
        Check(aclrtSetDevice(deviceId_), "aclrtSetDevice");
        deviceSet_ = true;
        Check(aclrtCreateStream(&stream_), "aclrtCreateStream");
    }

    template <typename T>
    aclTensor* CreateTensor(const std::vector<T>& host, const std::vector<int64_t>& shape, aclDataType dtype)
    {
        const int64_t elements = Numel(shape);
        if (static_cast<uint64_t>(elements) != host.size() ||
            host.size() > std::numeric_limits<size_t>::max() / sizeof(T) || aclDataTypeSize(dtype) != sizeof(T)) {
            throw std::invalid_argument("tensor shape, storage or dtype size mismatch");
        }
        const size_t bytes = host.size() * sizeof(T);
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t i = shape.size(); i > 1; --i) {
            strides[i - 2] = strides[i - 1] * shape[i - 1];
        }
        tensors_.push_back(std::make_unique<TensorResource>());
        auto& tensor = *tensors_.back();
        Check(aclrtMalloc(&tensor.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(tensor)");
        Check(aclrtMemcpy(tensor.device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy");
        tensor.descriptor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                            shape.data(), shape.size(), tensor.device);
        Check(tensor.descriptor == nullptr ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS, "aclCreateTensor");
        return tensor.descriptor;
    }

    aclOpExecutor** ExecutorAddress() { return &executor_; }

    aclOpExecutor* PrepareExecutor()
    {
        Check(aclSetAclOpExecutorRepeatable(executor_), "aclSetAclOpExecutorRepeatable");
        return executor_;
    }

    void* AllocateWorkspace(uint64_t bytes)
    {
        if (bytes > 0) {
            Check(aclrtMalloc(&workspace_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(workspace)");
        }
        return workspace_;
    }

    aclrtStream StreamForLaunch()
    {
        pending_ = true;
        return stream_;
    }

    void Synchronize()
    {
        Check(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
        pending_ = false;
    }

private:
    struct TensorResource {
        void* device = nullptr;
        aclTensor* descriptor = nullptr;
    };
    void Record(int result, const char* operation)
    {
        if (result != ACL_SUCCESS) {
            std::cerr << operation << " failed: " << result << std::endl;
            if (status_ == ACL_SUCCESS) {
                status_ = result;
            }
        }
    }
    int& status_;
    int32_t deviceId_ = 0;
    bool initialized_ = false;
    bool deviceSet_ = false;
    bool pending_ = false;
    aclrtStream stream_ = nullptr;
    aclOpExecutor* executor_ = nullptr;
    void* workspace_ = nullptr;
    std::vector<std::unique_ptr<TensorResource>> tensors_;
};
} // namespace SwigluExample

using SwigluExample::CheckHardwareSupport;
using SwigluExample::Numel;

int main()
{
    int status = ACL_SUCCESS;
    {
        SwigluExample::Session session(status);
        try {
            session.Init(0);
            if (!CheckHardwareSupport("SwigluGroupQuantV2")) {
                std::cout << "\n=== Test SKIPPED (hardware not supported) ===" << std::endl;
                return ACL_SUCCESS;
            }
            const std::vector<int64_t> xShape{64, 128};
            const std::vector<int64_t> yShape{64, 64};
            const std::vector<int64_t> scaleShape{64, 1, 2};
            std::vector<uint16_t> xHost(Numel(xShape), 0x3C00); // FP16 1.0
            std::vector<uint8_t> yHost(Numel(yShape), 0);
            std::vector<uint8_t> scaleHost(Numel(scaleShape), 0);
            std::vector<uint16_t> originHost(Numel(yShape), 0);

            aclTensor* x = session.CreateTensor(xHost, xShape, ACL_FLOAT16);
            aclTensor* y = session.CreateTensor(yHost, yShape, ACL_FLOAT8_E4M3FN);
            aclTensor* scale = session.CreateTensor(scaleHost, scaleShape, ACL_FLOAT8_E8M0);
            aclTensor* origin = session.CreateTensor(originHost, yShape, ACL_FLOAT16);

            uint64_t workspaceSize = 0;
            auto ret = aclnnSwigluGroupQuantV2GetWorkspaceSize(x, nullptr, nullptr, nullptr, ACL_FLOAT8_E4M3FN, 5, 32,
                                                               true, 7.0, 15.0, true, 1.702, 1.0, y, scale, origin,
                                                               &workspaceSize, session.ExecutorAddress());
            session.Check(ret, "aclnnSwigluGroupQuantV2");
            aclOpExecutor* executor = session.PrepareExecutor();
            void* workspace = session.AllocateWorkspace(workspaceSize);
            ret = aclnnSwigluGroupQuantV2(workspace, workspaceSize, executor, session.StreamForLaunch());
            session.Check(ret, "aclnnSwigluGroupQuantV2");
            session.Synchronize();

        } catch (const std::exception& error) {
            std::cerr << error.what() << std::endl;
            if (status == ACL_SUCCESS) {
                status = ACL_ERROR_INVALID_PARAM;
            }
        }
    }
    if (status != ACL_SUCCESS) {
        return 1;
    }
    std::cout << "aclnnSwigluGroupQuantV2 example succeeded" << std::endl;
    return 0;
}
```
