# SwigluGroupQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/9.2.0/activation/swiglu_group_quant)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     ×    |
|  <term>Atlas A2系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

### 接口功能

SwigluGroupQuant算子实现SwiGLU激活函数与分组量化融合计算。支持五种量化模式：

- **quant_mode=0**: Block Quant（FP8块量化，固定128元素分组）
- **quant_mode=1**: MX Quant（FP8 MX量化，固定32元素分组）
- **quant_mode=2**: HiFp8 Static Quant（HiFp8静态量化）
- **quant_mode=3**: HiFp8 Dynamic Quant（HiFp8动态量化）
- **quant_mode=5**: MX Quant V2（变体Clipped SwiGLU + CuBALS MX FP8量化，固定32元素分组）

### 计算公式

**基础计算流程**

```txt
步骤一：GroupIndex处理（可选）→ 计算real_bs
步骤二：输入切分（仅处理前real_bs行）
步骤三：Clamp处理（可选，仅处理前real_bs行）
步骤四：SwiGLU激活 / 变体SwiGLU激活（仅处理前real_bs行）
步骤五：Weight加权（可选，仅处理前real_bs行）
步骤六：量化计算（仅处理前real_bs行）
```

<details>
<summary><strong>步骤一：GroupIndex处理（可选）</strong></summary>

当提供`group_index`时，用于动态计算实际处理的token数量：

$$
\text{group\_sum} = \sum_{g=0}^{G-1} \text{group\_index}[g]
$$

$$
\text{real\_bs} = \min(\text{group\_sum}, N)
$$

其中：

- $G$为MoE专家分组数
- $N$为输入张量的第一维（预设batch size）
- 后续所有步骤仅处理前$\text{real\_bs}$行数据

**MoE场景说明**：在MoE推理中，不同专家可能处理不同数量的token，group_index允许动态调整处理范围，避免处理空数据。

</details>

<details>
<summary><strong>步骤二：输入切分</strong></summary>

输入张量 $\mathbf{x} \in \mathbb{R}^{N \times D}$ 沿最后一维切分为两部分：

$$
\mathbf{x}_0[n, d] = \mathbf{x}[n, d], \quad d \in [0, D/2)
$$

$$
\mathbf{x}_1[n, d] = \mathbf{x}[n, d + D/2], \quad d \in [0, D/2)
$$

</details>

<details>
<summary><strong>步骤三：Clamp处理（可选）</strong></summary>

当`clamp_limit > 0`时，对输入进行限制：

$$
\mathbf{x}_0'[n, d] = \min(\mathbf{x}_0[n, d], c)
$$

$$
\mathbf{x}_1'[n, d] = \min(\max(\mathbf{x}_1[n, d], -c), c)
$$

其中$c$为`clamp_limit`。

**Clamp的作用**：

- $\mathbf{x}_0$（门控分支）限制为正值范围$[0, c]$，防止sigmoid梯度消失
- $\mathbf{x}_1$（线性分支）限制为对称范围$[-c, c]$，防止数值溢出

</details>

<details>
<summary><strong>步骤四：SwiGLU激活 / 变体SwiGLU激活</strong></summary>

### 1. 标准SwiGLU（quant_mode=0/1/2/3）

逐元素计算：

$$
\mathbf{y}_{\text{origin}}[t,h]
=\operatorname{SiLU}(\mathbf{x}_0'[t,h])\cdot\mathbf{x}_1'[t,h]
$$

其中：

$$
\operatorname{SiLU}(z)=z\cdot\sigma(z)=\frac{z}{1+e^{-z}}
$$

因此：

$$
\mathbf{y}_{\text{origin}}[t,h]
=\frac{\mathbf{x}_0'[t,h]}{1+e^{-\mathbf{x}_0'[t,h]}}\cdot\mathbf{x}_1'[t,h]
$$

### 2. 变体Clipped SwiGLU（quant_mode=5）

`quant_mode=5`支持`alpha`和`bias`，激活公式为：

$$
\mathbf{y}_{\text{origin}}[t,h]
=\mathbf{x}_0'[t,h]\cdot
\sigma\!\left(\alpha\mathbf{x}_0'[t,h]\right)\cdot
\left(\mathbf{x}_1'[t,h]+\beta\right)
$$

其中：

- $\alpha$对应属性`alpha`，默认值为$1.0$；
- $\beta$对应属性`bias`，默认值为$0.0$；
- Clamp在`alpha`和`bias`参与激活计算之前执行。

等价的分步计算为：

$$
\begin{aligned}
t_1[t,h]&=-\alpha\mathbf{x}_0'[t,h] \\
t_2[t,h]&=e^{t_1[t,h]} \\
t_3[t,h]&=1+t_2[t,h] \\
t_4[t,h]&=\frac{\mathbf{x}_0'[t,h]}{t_3[t,h]} \\
t_5[t,h]&=\mathbf{x}_1'[t,h]+\beta \\
\mathbf{y}_{\text{origin}}[t,h]&=t_4[t,h]\cdot t_5[t,h]
\end{aligned}
$$

当$\alpha=1.0$、$\beta=0.0$时，变体公式退化为标准SwiGLU。

**yOrigin输出**：当`output_origin=true`时，`yOrigin`保存上述激活结果，即**Weight加权前**的SwiGLU结果；对于`quant_mode=5`，其中已包含Clamp、`alpha`和`bias`的作用。

</details>

<details>
<summary><strong>步骤五：Weight加权（可选）</strong></summary>

当提供`weight`时，对SwiGLU输出进行加权：

$$
\mathbf{y}_{\text{weighted}}[n, d] = \mathbf{y}_{\text{swiglu}}[n, d] \cdot w[n]
$$

其中$w[n]$为第$n$个token的weight值。

**MoE场景**：weight来自专家路由器的softmax输出，表示该token对当前专家的权重。

</details>

<details>
<summary><strong>步骤六：量化计算</strong></summary>

<details>
<summary><strong>quant_mode=0 (Block Quant)</strong></summary>

**分组划分**：将输出沿最后一维按128元素为一组划分：

$$
\mathbf{y} = [\mathbf{g}_0, \mathbf{g}_1, \ldots, \mathbf{g}_K], \quad K = \lceil D/2 / 128 \rceil
$$

每个组$\mathbf{g}_i \in \mathbb{R}^{N \times 128}$。

**非有限值屏蔽与绝对值计算**：

$$
\begin{aligned}
\mathbf{z}[n, j] &= \mathbf{y}_{\text{weighted}}[n, j] \cdot 0 \quad \text{(生成零张量)} \\
\mathbf{m}_{\text{finite}}[n, j] &= (\mathbf{z}[n, j] = \mathbf{z}[n, j]) \\
\mathbf{y}_{\text{abs}}[n, j] &=
\begin{cases}
|\mathbf{y}_{\text{weighted}}[n, j]|, & \mathbf{m}_{\text{finite}}[n, j] \\
0, & \text{otherwise}
\end{cases}
\end{aligned}
$$

**屏蔽原理**：NaN的特性是`NaN != NaN`；同时`Inf * 0`也会得到NaN，因此该步骤在计算amax时屏蔽NaN和Inf。

**Scale计算**：

对于第$i$个组（包含128个连续元素）：

$$
a_i = \max_{j=0}^{127} \mathbf{y}_{\text{abs}}[j]
$$

$$
\hat{a}_i = \max(a_i, 10^{-4})
$$

$$
s_i^{\text{raw}} = \frac{\hat{a}_i}{M_{\text{fp8}}}
$$

其中$M_{\text{fp8}}$取值：

- FP8 E4M3FN：$M_{\text{fp8}} = 448.0$
- FP8 E5M2：$M_{\text{fp8}} = 57344.0$

**Scale输出与InvScale计算**：

当`round_scale=false`时：

$$
s_i = s_i^{\text{raw}}, \quad \text{InvScale}_i = \frac{M_{\text{fp8}}}{\hat{a}_i} = \frac{1}{s_i}
$$

当`round_scale=true`时，将scale向上取整到2的幂：

$$
e_i = \lceil \log_2(s_i^{\text{raw}}) \rceil
$$

$$
s_i = 2^{e_i}, \quad \text{InvScale}_i = 2^{-e_i}
$$

其中$s_i$写入FLOAT32类型的scale输出。

**量化计算**：

$$
\mathbf{y}_{\text{scaled}}[n, j] = \mathbf{y}_{\text{weighted}}[n, j] \cdot \text{InvScale}_i, \quad j \in \text{group } i
$$

若 $\mathbf{y}_{\text{scaled}}[n,j]$ 为NaN或Inf，实现会使用原始 $\mathbf{y}_{\text{weighted}}[n,j]$ 作为FP8 cast输入：

$$
\mathbf{y}_{\text{cast\_in}}[n, j] =
\begin{cases}
\mathbf{y}_{\text{scaled}}[n, j], & \mathbf{y}_{\text{scaled}}[n, j] \text{ is finite} \\
\mathbf{y}_{\text{weighted}}[n, j], & \text{otherwise}
\end{cases}
$$

$$
\mathbf{y}_{\text{quant}}[n, j] = \text{cast\_fp8\_rint}(\mathbf{y}_{\text{cast\_in}}[n, j])
$$

其中`cast_fp8_rint`为FP32到FP8的类型转换，采用**RINT（就近舍入）**模式。

</details>

<details>
<summary><strong>quant_mode=1 (MX Quant)</strong></summary>

**MX量化原理**：采用**E8M0 Scale** + **FP8 Data**的组合。

**分组方式**：每**32元素**为一组：

$$
\mathbf{y} = [\mathbf{g}_0, \mathbf{g}_1, \ldots, \mathbf{g}_K], \quad \mathbf{g}_i \in \mathbb{R}^{32}
$$

**Amax计算**：

$$
a_i = \max_{j=0}^{31} |\mathbf{g}_i[j]|
$$

$$
\hat{a}_i = \max(a_i, 10^{-4})
$$

**原始Scale计算**：

$$
s_i^{\text{raw}} = \frac{\hat{a}_i}{M_{\text{fp8}}}
$$

其中$M_{\text{fp8}}$取值：

- FP8 E4M3FN：$M_{\text{fp8}} = 448.0$
- FP8 E5M2：$M_{\text{fp8}} = 57344.0$

quant_mode=1仅支持`round_scale=true`，将原始scale向上取整到2的幂：

$$
e_i = \lceil \log_2(s_i^{\text{raw}}) \rceil
$$

等价于基于FP32位模式计算：

$$
e_i = E(s_i^{\text{raw}}) - 127 + \mathbf{1}_{\text{mantissa}(s_i^{\text{raw}}) \ne 0}
$$

**E8M0 Scale编码**：

$$
s_i^{\text{e8m0}} = e_i + 127
$$

其中$s_i^{\text{e8m0}}$写入FLOAT8_E8M0类型的scale输出，表示的实际scale值为$2^{e_i}$。

**InvScale计算**：

$$
\text{InvScale}_i = 2^{-e_i}
$$

**量化计算**：

$$
\mathbf{y}_{\text{quant}}[j] = \text{cast\_fp8\_rint}\left(\mathbf{y}_{\text{weighted}}[j] \cdot \text{InvScale}_i\right), \quad j \in \text{group } i
$$

</details>

<details>
<summary><strong>quant_mode=2 (HiFp8 Static Quant)</strong></summary>

**静态量化说明**：使用预先提供的`invScale`对加权后的SwiGLU输出进行缩放量化。

**情况1：无GroupIndex**（groupIndex为空）：

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\mathbf{y}_{\text{weighted}}[n, d] \cdot \text{invScale}[0]\right), \quad n \in [0, N), \quad d \in [0, D/2)
$$

其中`hif8_cast`为HiFloat8类型转换函数。

**情况2：有GroupIndex**（groupIndex非空）：

设$G$为MoE专家分组数，$\text{groupIndex}[g]$表示第$g$个专家处理的token数量。

计算每个group的起止索引：

$$
\text{start}^{(0)} = 0, \quad \text{end}^{(g)} = \sum_{k=0}^{g} \text{groupIndex}[k], \quad \text{start}^{(g)} = \text{end}^{(g-1)}
$$

对于第$g$个group，使用对应的缩放因子$\text{invScale}[g]$进行量化：

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\mathbf{y}_{\text{weighted}}[n, d] \cdot \text{invScale}[g]\right), \quad n \in [\text{start}^{(g)}, \text{end}^{(g)}), \quad d \in [0, D/2)
$$

**MoE场景说明**：在MoE推理中，不同专家处理不同数量的token，groupIndex用于标识每个专家处理的token范围，invScale为每个专家预先计算的静态缩放因子。

</details>

<details>
<summary><strong>quant_mode=3 (HiFp8 Dynamic Quant)</strong></summary>

**动态量化说明**：根据加权后的SwiGLU输出动态计算缩放因子进行量化。

**情况1：无GroupIndex**（groupIndex为空）：

计算全局绝对值最大值：

$$
a_{\max} = \max\left(\max_{n \in [0, N), d \in [0, D/2)} |\mathbf{y}_{\text{weighted}}[n, d]|, \epsilon\right)
$$

其中$\epsilon$为数值稳定性常数。

计算缩放因子：

$$
s = \frac{a_{\max}}{M_{\text{hif8}}}
$$

其中$M_{\text{hif8}}$为`dstTypeMax`，表示HiFloat8类型的最大有限值。

量化计算：

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\frac{\mathbf{y}_{\text{weighted}}[n, d]}{s}\right), \quad n \in [0, N), \quad d \in [0, D/2)
$$

其中`hif8_cast`为HiFloat8类型转换函数。

**情况2：有GroupIndex**（groupIndex非空）：

设$G$为MoE专家分组数，$\text{groupIndex}[g]$表示第$g$个专家处理的token数量。

计算每个group的起止索引：

$$
\text{start}^{(0)} = 0, \quad \text{end}^{(g)} = \sum_{k=0}^{g} \text{groupIndex}[k], \quad \text{start}^{(g)} = \text{end}^{(g-1)}
$$

对于第$g$个group，提取对应的数据：

$$
\mathbf{y}^{(g)} = \mathbf{y}_{\text{weighted}}[\text{start}^{(g)}:\text{end}^{(g)}, :]
$$

计算该group的绝对值最大值：

$$
a_{\max}^{(g)} = \max\left(\max_{n \in [\text{start}^{(g)}, \text{end}^{(g)}), d \in [0, D/2)} |\mathbf{y}^{(g)}[n, d]|, \epsilon\right)
$$

计算该group的缩放因子：

$$
s^{(g)} = \frac{a_{\max}^{(g)}}{M_{\text{hif8}}}
$$

对该group进行量化：

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\frac{\mathbf{y}_{\text{weighted}}[n, d]}{s^{(g)}}\right), \quad n \in [\text{start}^{(g)}, \text{end}^{(g)}), \quad d \in [0, D/2)
$$

**MoE场景说明**：在MoE推理中，不同专家处理不同数量的token，groupIndex用于标识每个专家处理的token范围，每个group独立计算缩放因子以适应不同数据分布。

</details>

<details>
<summary><strong>quant_mode=5 (MxQuant CuBALS)</strong></summary>

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

</details>

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 100px">
  <col style="width: 420px">
  <col style="width: 240px">
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
      <td>SwiGLU输入。quantMode为5时shape仅支持[T,D]二维；其他模式shape为[...,D]，维度为2-8维（quantMode为1时为2-7维）。quantMode为5时D必须大于等于64且能被64整除；其他模式的非空输入D必须大于等于256且能被256整除。quantMode为0或1时支持空Tensor；quantMode为2、3或5时不支持空Tensor。quantMode为0、1或5时仅支持FLOAT16、BFLOAT16；quantMode为2或3时支持FLOAT、FLOAT16、BFLOAT16。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入（可选）</td>
      <td>MOE权重张量，用于SwiGLU输出的加权计算。quantMode为0或1时支持空Tensor；quantMode为2、3或5时不支持空Tensor。不为空时，quantMode=5支持FLOAT16、BFLOAT16、FLOAT32，其他模式仅支持FLOAT32；维度为1-8维，元素个数需等于x除最后一维外的元素个数之积。</td>
      <td>quantMode=5：FLOAT16、BFLOAT16、FLOAT32；其他模式：FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>输入（可选）</td>
      <td>count模式的group token数。quantMode为0或1时不支持单独为空Tensor；quantMode为2或3不支持空Tensor。不为空时，数据类型为INT64，shape为[G]。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入（可选）</td>
      <td>quantMode=2时静态量化输入的invScale张量。仅quant_mode为2时使用。groupIndex存在的话，shape=[G]，不存在的话shape=[1]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>属性</td>
      <td>目标量化类型。quantMode为0、1或5时生效。支持取值35、36，分别表示FLOAT8_E5M2、FLOAT8_E4M3FN。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>属性</td>
      <td>量化模式。支持取值0、1、2、3、5。0表示Block FP8模式。1表示原MX模式。2表示HIFP8静态量化模式。3表示HIFP8动态量化模式。5表示MX V2模式。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>属性</td>
      <td>量化块大小。0表示使用当前量化模式的默认block大小。quantMode为0时支持0或128；quantMode为1或5时支持0或32；quantMode为2或3时该参数不生效。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_scale</td>
      <td>属性</td>
      <td>是否将scale取整为2的幂。quantMode为1或5时必须为true。quantMode为2或3时该参数不生效。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clamp_limit</td>
      <td>属性</td>
      <td>SwiGLU计算前的clamp阈值。-1.0表示不启用clamp。启用clamp时，clampLimit必须大于0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>属性</td>
      <td>目标量化类型的最大有限值。仅quantMode为3时，该参数生效。默认值为15.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_origin</td>
      <td>属性</td>
      <td>是否输出weight加权前的SwiGLU结果。true表示输出有效yOrigin，为false时yOrigin输出无效。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>量化输出。quantMode为0、1或5时数据类型与dstType一致；quantMode为2或3时数据类型为HIFLOAT8。shape均为[...,D/2]。quantMode为0或1时支持空Tensor；quantMode为2、3或5时不支持空Tensor。</td>
      <td>HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_scale</td>
      <td>输出</td>
      <td>量化scale输出。quantMode为0时shape为[...,ceil((D/2)/128)]，数据类型为FLOAT32。quantMode为1或5时shape为[...,ceil(ceil((D/2)/32)/2),2]，数据类型为FLOAT8_E8M0。quantMode为2或3时，无groupIndex时shape为[1]，有groupIndex时shape为[G]，数据类型为FLOAT32。quantMode为0或1时支持空Tensor；quantMode为2、3或5时不支持空Tensor。</td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_origin</td>
      <td>输出</td>
      <td>weight加权前的SwiGLU结果。shape为[...,D/2]。数据类型需与x一致。不支持空指针。quantMode为0或1时支持空Tensor；quantMode为2或3不支持空Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 确定性计算：aclnnSwigluGroupQuant默认确定性实现。
- quantMode为0时，仅支持FP8输出，blockSize支持0或128。
- quantMode为1时，支持FP8输出，blockSize支持0或32，roundScale必须为true。
- quantMode为5时，x仅支持二维[T,D]；支持FP8输出，blockSize支持0或32，roundScale必须为true；不支持groupIndex和scale。
  clampLimit必须为有限值，取-1.0表示关闭Clamp，否则必须大于0。
- quantMode为2或3时，支持HIFP8量化输出，dstType, blockSize和roundScale不生效。输入x的维度为[T, D]或[B, S, D]，需满足以下规格约束：

  | 规格项 | 规格 | 规格说明 |
  | :--- | :--- | :--- |
  | B | 1~31 | - |
  | S | 0~128K | - |
  | D/2 | 512, 768, 1024, 1536, 1792, 2048, 2560, 4096 | - |
  | dstTypeMax | 15, 56, 224, 32768 | - |

- yScale的数据类型必须与quantMode匹配：Block FP8为FLOAT32，MX为FLOAT8_E8M0，HIFP8为FLOAT32。
- groupIndex中的元素值须大于等于0。

## 调用说明

|调用方式|调用样例|说明|
|:-------|:-------|:---|
|aclnn调用|[test_aclnn_swiglu_group_quant](./examples/test_aclnn_swiglu_group_quant.cpp)|通过[aclnnSwigluGroupQuant](./docs/aclnnSwigluGroupQuant.md)接口调用SwigluGroupQuant算子。|
|aclnn V2调用|[test_aclnn_swiglu_group_quant_v2](./examples/test_aclnn_swiglu_group_quant_v2.cpp)|通过[aclnnSwigluGroupQuantV2](./docs/aclnnSwigluGroupQuantV2.md)接口调用SwigluGroupQuant算子的quant_mode=5。|
|图模式调用|-|通过[算子IR](./op_graph/swiglu_group_quant_proto.h)构图方式调用SwigluGroupQuant算子。|
