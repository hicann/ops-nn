# SwigluGroupQuant

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

### Description:

The SwigluGroupQuant operator implements the fused computation of the SwiGLU activation function and group quantization. Five quantization modes are supported:

- **quant_mode=0**: Block Quant (FP8 block quantization, fixed 128-element groups)
- **quant_mode=1**: MX Quant (FP8 MX quantization, fixed 32-element groups)
- **quant_mode=2**: HiFp8 Static Quant (HiFp8 static quantization)
- **quant_mode=3**: HiFp8 Dynamic Quant (HiFp8 dynamic quantization)
- **quant_mode=5**: MX Quant V2 (variant Clipped SwiGLU + CuBALS MX FP8 quantization, fixed 32-element groups)

### Formula:

**Basic computation flow**

```txt
Step 1: GroupIndex processing (optional) → Compute real_bs
Step 2: Input splitting (processes only the first real_bs rows)
Step 3: Clamp processing (optional, processes only the first real_bs rows)
Step 4: SwiGLU activation / variant SwiGLU activation (processes only the first real_bs rows)
Step 5: Weight weighting (optional, processes only the first real_bs rows)
Step 6: Quantization (processes only the first real_bs rows)
```

<details>
<summary><strong>Step 1: GroupIndex processing (optional)</strong></summary>

When `group_index` is provided, it dynamically computes the number of tokens to actually process:

$$
\text{group\_sum} = \sum_{g=0}^{G-1} \text{group\_index}[g]
$$

$$
\text{real\_bs} = \min(\text{group\_sum}, N)
$$

where:

- $G$ is the number of MoE expert groups
- $N$ is the first dimension of the input tensor (preset batch size)
- All subsequent steps process only the first $\text{real\_bs}$ rows of data

**MoE scenario**: In MoE inference, different experts may process different numbers of tokens. group_index allows dynamic adjustment of the processing range to avoid processing empty data.

</details>

<details>
<summary><strong>Step 2: Input splitting</strong></summary>

The input tensor $\mathbf{x} \in \mathbb{R}^{N \times D}$ is split along the last dimension into two parts:

$$
\mathbf{x}_0[n, d] = \mathbf{x}[n, d], \quad d \in [0, D/2)
$$

$$
\mathbf{x}_1[n, d] = \mathbf{x}[n, d + D/2], \quad d \in [0, D/2)
$$

</details>

<details>
<summary><strong>Step 3: Clamp processing (optional)</strong></summary>

When `clamp_limit > 0`, the inputs are clamped:

$$
\mathbf{x}_0'[n, d] = \min(\mathbf{x}_0[n, d], c)
$$

$$
\mathbf{x}_1'[n, d] = \min(\max(\mathbf{x}_1[n, d], -c), c)
$$

where $c$ is `clamp_limit`.

**Purpose of Clamp**:

- $\mathbf{x}_0$ (gating branch) is limited to the positive range $[0, c]$ to prevent gradient vanishing of sigmoid
- $\mathbf{x}_1$ (linear branch) is limited to the symmetric range $[-c, c]$ to prevent numeric overflow

</details>

<details>
<summary><strong>Step 4: SwiGLU activation / variant SwiGLU activation</strong></summary>

### 1. Standard SwiGLU (quant_mode=0/1/2/3)

Element-wise computation:

$$
\mathbf{y}_{\text{origin}}[t,h]
=\operatorname{SiLU}(\mathbf{x}_0'[t,h])\cdot\mathbf{x}_1'[t,h]
$$

where:

$$
\operatorname{SiLU}(z)=z\cdot\sigma(z)=\frac{z}{1+e^{-z}}
$$

Therefore:

$$
\mathbf{y}_{\text{origin}}[t,h]
=\frac{\mathbf{x}_0'[t,h]}{1+e^{-\mathbf{x}_0'[t,h]}}\cdot\mathbf{x}_1'[t,h]
$$

### 2. Variant Clipped SwiGLU (quant_mode=5)

`quant_mode=5` supports `alpha` and `bias`. The activation formula is:

$$
\mathbf{y}_{\text{origin}}[t,h]
=\mathbf{x}_0'[t,h]\cdot
\sigma\!\left(\alpha\mathbf{x}_0'[t,h]\right)\cdot
\left(\mathbf{x}_1'[t,h]+\beta\right)
$$

where:

- $\alpha$ corresponds to the attribute `alpha`, with a default value of $1.0$;
- $\beta$ corresponds to the attribute `bias`, with a default value of $0.0$;
- Clamp is performed before `alpha` and `bias` participate in the activation computation.

The equivalent step-by-step computation is:

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

When $\alpha=1.0$ and $\beta=0.0$, the variant formula degenerates to standard SwiGLU.

**yOrigin output**: When `output_origin=true`, `yOrigin` stores the preceding activation result, that is, the SwiGLU result **before Weight weighting**; for `quant_mode=5`, it already includes the effects of Clamp, `alpha`, and `bias`.

</details>

<details>
<summary><strong>Step 5: Weight weighting (optional)</strong></summary>

When `weight` is provided, the SwiGLU output is weighted:

$$
\mathbf{y}_{\text{weighted}}[n, d] = \mathbf{y}_{\text{swiglu}}[n, d] \cdot w[n]
$$

where $w[n]$ is the weight value of the n-th token.

**MoE scenario**: weight comes from the softmax output of the expert router and represents the weight of this token for the current expert.

</details>

<details>
<summary><strong>Step 6: Quantization</strong></summary>

<details>
<summary><strong>quant_mode=0 (Block Quant)</strong></summary>

**Group division**: The output is divided along the last dimension into groups of 128 elements:

$$
\mathbf{y} = [\mathbf{g}_0, \mathbf{g}_1, \ldots, \mathbf{g}_K], \quad K = \lceil D/2 / 128 \rceil
$$

Each group is $\mathbf{g}_i \in \mathbb{R}^{N \times 128}$.

**Non-finite value masking and absolute value computation**:

$$
\begin{aligned}
\mathbf{z}[n, j] &= \mathbf{y}_{\text{weighted}}[n, j] \cdot 0 \quad \text{(generate a zero tensor)} \\
\mathbf{m}_{\text{finite}}[n, j] &= (\mathbf{z}[n, j] = \mathbf{z}[n, j]) \\
\mathbf{y}_{\text{abs}}[n, j] &=
\begin{cases}
|\mathbf{y}_{\text{weighted}}[n, j]|, & \mathbf{m}_{\text{finite}}[n, j] \\
0, & \text{otherwise}
\end{cases}
\end{aligned}
$$

**Masking principle**: NaN has the property `NaN != NaN`; meanwhile `Inf * 0` also yields NaN, so this step masks NaN and Inf when computing amax.

**Scale computation**:

For the i-th group (containing 128 consecutive elements):

$$
a_i = \max_{j=0}^{127} \mathbf{y}_{\text{abs}}[j]
$$

$$
\hat{a}_i = \max(a_i, 10^{-4})
$$

$$
s_i^{\text{raw}} = \frac{\hat{a}_i}{M_{\text{fp8}}}
$$

where $M_{\text{fp8}}$ takes the following values:

- FP8 E4M3FN: $M_{\text{fp8}} = 448.0$
- FP8 E5M2: $M_{\text{fp8}} = 57344.0$

**Scale output and InvScale computation**:

When `round_scale=false`:

$$
s_i = s_i^{\text{raw}}, \quad \text{InvScale}_i = \frac{M_{\text{fp8}}}{\hat{a}_i} = \frac{1}{s_i}
$$

When `round_scale=true`, the scale is rounded up to a power of 2:

$$
e_i = \lceil \log_2(s_i^{\text{raw}}) \rceil
$$

$$
s_i = 2^{e_i}, \quad \text{InvScale}_i = 2^{-e_i}
$$

where $s_i$ is written to the FLOAT32 scale output.

**Quantization computation**:

$$
\mathbf{y}_{\text{scaled}}[n, j] = \mathbf{y}_{\text{weighted}}[n, j] \cdot \text{InvScale}_i, \quad j \in \text{group } i
$$

If $\mathbf{y}_{\text{scaled}}[n,j]$ is NaN or Inf, the implementation uses the original $\mathbf{y}_{\text{weighted}}[n,j]$ as the FP8 cast input:

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

where `cast_fp8_rint` is the type conversion from FP32 to FP8, using the **RINT (round to nearest)** mode.

</details>

<details>
<summary><strong>quant_mode=1 (MX Quant)</strong></summary>

**MX quantization principle**: adopts the combination of **E8M0 Scale** + **FP8 Data**.

**Grouping**: every **32 elements** form one group:

$$
\mathbf{y} = [\mathbf{g}_0, \mathbf{g}_1, \ldots, \mathbf{g}_K], \quad \mathbf{g}_i \in \mathbb{R}^{32}
$$

**Amax computation**:

$$
a_i = \max_{j=0}^{31} |\mathbf{g}_i[j]|
$$

$$
\hat{a}_i = \max(a_i, 10^{-4})
$$

**Raw scale computation**:

$$
s_i^{\text{raw}} = \frac{\hat{a}_i}{M_{\text{fp8}}}
$$

where $M_{\text{fp8}}$ takes the following values:

- FP8 E4M3FN: $M_{\text{fp8}} = 448.0$
- FP8 E5M2: $M_{\text{fp8}} = 57344.0$

quant_mode=1 supports only `round_scale=true`, which rounds the raw scale up to a power of 2:

$$
e_i = \lceil \log_2(s_i^{\text{raw}}) \rceil
$$

The equivalent computation based on the FP32 bit pattern is:

$$
e_i = E(s_i^{\text{raw}}) - 127 + \mathbf{1}_{\text{mantissa}(s_i^{\text{raw}}) \ne 0}
$$

**E8M0 scale encoding**:

$$
s_i^{\text{e8m0}} = e_i + 127
$$

where $s_i^{\text{e8m0}}$ is written to the FLOAT8_E8M0 scale output, and the actual scale value represented is $2^{e_i}$.

**InvScale computation**:

$$
\text{InvScale}_i = 2^{-e_i}
$$

**Quantization computation**:

$$
\mathbf{y}_{\text{quant}}[j] = \text{cast\_fp8\_rint}\left(\mathbf{y}_{\text{weighted}}[j] \cdot \text{InvScale}_i\right), \quad j \in \text{group } i
$$

</details>

<details>
<summary><strong>quant_mode=2 (HiFp8 Static Quant)</strong></summary>

**Static quantization description**: uses the pre-provided `invScale` to perform scaled quantization on the weighted SwiGLU output.

**Case 1: no GroupIndex** (groupIndex is empty):

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\mathbf{y}_{\text{weighted}}[n, d] \cdot \text{invScale}[0]\right), \quad n \in [0, N), \quad d \in [0, D/2)
$$

where `hif8_cast` is the HiFloat8 type conversion function.

**Case 2: with GroupIndex** (groupIndex is not empty):

Let $G$ be the number of MoE expert groups, and $\text{groupIndex}[g]$ denote the number of tokens processed by the g-th expert.

Compute the start and end indices of each group:

$$
\text{start}^{(0)} = 0, \quad \text{end}^{(g)} = \sum_{k=0}^{g} \text{groupIndex}[k], \quad \text{start}^{(g)} = \text{end}^{(g-1)}
$$

For the g-th group, quantization uses the corresponding scaling factor $\text{invScale}[g]$:

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\mathbf{y}_{\text{weighted}}[n, d] \cdot \text{invScale}[g]\right), \quad n \in [\text{start}^{(g)}, \text{end}^{(g)}), \quad d \in [0, D/2)
$$

**MoE scenario**: In MoE inference, different experts process different numbers of tokens. groupIndex identifies the token range processed by each expert, and invScale is the static scaling factor precomputed for each expert.

</details>

<details>
<summary><strong>quant_mode=3 (HiFp8 Dynamic Quant)</strong></summary>

**Dynamic quantization description**: computes the scaling factor dynamically from the weighted SwiGLU output for quantization.

**Case 1: no GroupIndex** (groupIndex is empty):

Compute the global maximum absolute value:

$$
a_{\max} = \max\left(\max_{n \in [0, N), d \in [0, D/2)} |\mathbf{y}_{\text{weighted}}[n, d]|, \epsilon\right)
$$

where $\epsilon$ is a constant for numerical stability.

Compute the scaling factor:

$$
s = \frac{a_{\max}}{M_{\text{hif8}}}
$$

where $M_{\text{hif8}}$ is `dstTypeMax`, representing the maximum finite value of the HiFloat8 type.

Quantization computation:

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\frac{\mathbf{y}_{\text{weighted}}[n, d]}{s}\right), \quad n \in [0, N), \quad d \in [0, D/2)
$$

where `hif8_cast` is the HiFloat8 type conversion function.

**Case 2: with GroupIndex** (groupIndex is not empty):

Let $G$ be the number of MoE expert groups, and $\text{groupIndex}[g]$ denote the number of tokens processed by the g-th expert.

Compute the start and end indices of each group:

$$
\text{start}^{(0)} = 0, \quad \text{end}^{(g)} = \sum_{k=0}^{g} \text{groupIndex}[k], \quad \text{start}^{(g)} = \text{end}^{(g-1)}
$$

For the g-th group, extract the corresponding data:

$$
\mathbf{y}^{(g)} = \mathbf{y}_{\text{weighted}}[\text{start}^{(g)}:\text{end}^{(g)}, :]
$$

Compute the maximum absolute value of this group:

$$
a_{\max}^{(g)} = \max\left(\max_{n \in [\text{start}^{(g)}, \text{end}^{(g)}), d \in [0, D/2)} |\mathbf{y}^{(g)}[n, d]|, \epsilon\right)
$$

Compute the scaling factor of this group:

$$
s^{(g)} = \frac{a_{\max}^{(g)}}{M_{\text{hif8}}}
$$

Quantize this group:

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\frac{\mathbf{y}_{\text{weighted}}[n, d]}{s^{(g)}}\right), \quad n \in [\text{start}^{(g)}, \text{end}^{(g)}), \quad d \in [0, D/2)
$$

**MoE scenario**: In MoE inference, different experts process different numbers of tokens. groupIndex identifies the token range processed by each expert, and each group computes its scaling factor independently to adapt to different data distributions.

</details>

<details>
<summary><strong>quant_mode=5 (MxQuant CuBALS)</strong></summary>

**MX quantization principle**: adopts the combination of **E8M0 Scale** + **FP8 Data**.

**Grouping**: every **32 elements** form one group; fewer than 32 elements are padded with 0 to participate in the computation:

$$
\mathbf{y} = [\mathbf{g}_0, \mathbf{g}_1, \ldots, \mathbf{g}_K], \quad \mathbf{g}_i \in \mathbb{R}^{32}
$$

**Amax computation** (CuBALS has no lower bound):

$$
a_i = \max_{j=0}^{31} |\mathbf{g}_i[j]|
$$

**Raw scale computation**:

$$
s_i^{\text{raw}} = \frac{a_i}{M_{\text{fp8}}}
$$

where $M_{\text{fp8}}$ takes the following values:

- FP8 E4M3FN: $M_{\text{fp8}} = 448.0$
- FP8 E5M2: $M_{\text{fp8}} = 57344.0$

**Exponent rounding up (CuBALS)**: extract the unbiased exponent $E_i$ and mantissa $M_i$ from $s_i^{\text{raw}}$.
To ensure that quantization does not overflow, the exponent is rounded up:

$$
E_i^{*} =
\begin{cases}
E_i + 1, & s_i^{\text{raw}} \text{ is a normal number, } E_i < 254 \text{ and } M_i > 0 \\
E_i + 1, & s_i^{\text{raw}} \text{ is a subnormal number and } M_i > 0.5 \\
E_i, & \text{otherwise}
\end{cases}
$$

$$
s_i = 2^{E_i^{*}}
$$

**E8M0 scale encoding**: $s_i$ is stored into FLOAT8_E8M0 as an unbiased exponent:

$$
s_i^{\text{e8m0}} = E_i^{*} + 127
$$

**All-zero block handling**: when $a_i = 0$, $s_i^{\text{e8m0}} = 0$ (no amax lower bound, consistent with the dual-axis operator).

**Quantization computation**:

$$
\mathbf{y}_{\text{quant}}[t, j] = \text{cast\_fp8\_rint}\!\left(\mathbf{y}_{\text{weighted}}[t, j] \cdot \frac{1}{s_i}\right),
\quad j \in \text{group } i
$$

where `cast_fp8_rint` is the type conversion from FP32 to FP8, using the **RINT (round to nearest)** mode.

</details>

</details>

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 100px">
  <col style="width: 420px">
  <col style="width: 240px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>SwiGLU input. When quantMode is 5, the shape supports only the 2D [T,D]; in other modes, the shape is [...,D] with 2 to 8 dimensions (2 to 7 dimensions when quantMode is 1). When quantMode is 5, D must be greater than or equal to 64 and divisible by 64; in other modes, D of a non-empty input must be greater than or equal to 256 and divisible by 256. When quantMode is 0 or 1, an empty tensor is supported; when quantMode is 2 or 3, an empty tensor is not supported. When quantMode is 0, 1, or 5, only FLOAT16 and BFLOAT16 are supported; when quantMode is 2 or 3, FLOAT, FLOAT16, and BFLOAT16 are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input (optional)</td>
      <td>MOE weight tensor, used for the weighting computation of the SwiGLU output. When quantMode is 0 or 1, an empty tensor is supported; when quantMode is 2 or 3, an empty tensor is not supported. When it is not empty, quantMode=5 supports FLOAT16, BFLOAT16, and FLOAT32, and other modes support only FLOAT32; the dimensions range from 1 to 8, and the number of elements must equal the product of the element counts of x over all dimensions except the last one.</td>
      <td>quantMode=5: FLOAT16, BFLOAT16, FLOAT32; other modes: FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>Input (optional)</td>
      <td>Number of group tokens in count mode. When quantMode is 0 or 1, it does not support being an empty tensor alone; when quantMode is 2 or 3, an empty tensor is not supported. When it is not empty, the data type is INT64 and the shape is [G].</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input (optional)</td>
      <td>invScale tensor for static quantization input when quantMode=2. It is used only when quant_mode is 2. If groupIndex exists, the shape is [G]; otherwise, the shape is [1].</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Attribute</td>
      <td>Target quantization type. It takes effect when quantMode is 0, 1, or 5. The supported values are 35 and 36, which represent FLOAT8_E5M2 and FLOAT8_E4M3FN respectively.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Attribute</td>
      <td>Quantization mode. The supported values are 0, 1, 2, 3, and 5. 0 indicates the Block FP8 mode. 1 indicates the original MX mode. 2 indicates the HIFP8 static quantization mode. 3 indicates the HIFP8 dynamic quantization mode. 5 indicates the MX V2 mode.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>Attribute</td>
      <td>Quantization block size. 0 indicates using the default block size of the current quantization mode. When quantMode is 0, 0 or 128 is supported; when quantMode is 1 or 5, 0 or 32 is supported; when quantMode is 2 or 3, this parameter does not take effect.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_scale</td>
      <td>Attribute</td>
      <td>Whether to round the scale to a power of 2. When quantMode is 1 or 5, it must be true. When quantMode is 2 or 3, this parameter does not take effect.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clamp_limit</td>
      <td>Attribute</td>
      <td>Clamp threshold before the SwiGLU computation. -1.0 indicates that clamp is disabled. When clamp is enabled, clampLimit must be greater than 0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>Attribute</td>
      <td>Maximum finite value of the target quantization type. This parameter takes effect only when quantMode is 3. The default value is 15.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_origin</td>
      <td>Attribute</td>
      <td>Whether to output the SwiGLU result before Weight weighting. true indicates a valid yOrigin output; when it is false, the yOrigin output is invalid.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Quantization output. When quantMode is 0, 1, or 5, the data type is the same as dstType; when quantMode is 2 or 3, the data type is HIFLOAT8. The shape is [...,D/2] in all cases. When quantMode is 0 or 1, an empty tensor is supported; when quantMode is 2 or 3, an empty tensor is not supported.</td>
      <td>HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_scale</td>
      <td>Output</td>
      <td>Quantization scale output. When quantMode is 0, the shape is [...,ceil((D/2)/128)] and the data type is FLOAT32. When quantMode is 1 or 5, the shape is [...,ceil(ceil((D/2)/32)/2),2] and the data type is FLOAT8_E8M0. When quantMode is 2 or 3, the shape is [1] without groupIndex and [G] with groupIndex, and the data type is FLOAT32. When quantMode is 0 or 1, an empty tensor is supported; when quantMode is 2 or 3, an empty tensor is not supported.</td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_origin</td>
      <td>Output</td>
      <td>SwiGLU result before Weight weighting. The shape is [...,D/2]. The data type must be consistent with x. A null pointer is not supported. When quantMode is 0 or 1, an empty tensor is supported; when quantMode is 2 or 3, an empty tensor is not supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Deterministic computation: aclnnSwigluGroupQuant is deterministic by default.
- When quantMode is 0, only FP8 output is supported, and blockSize supports 0 or 128.
- When quantMode is 1, FP8 output is supported, blockSize supports 0 or 32, and roundScale must be true.
- When quantMode is 5, x supports only the 2D [T,D]; FP8 output is supported, blockSize supports 0 or 32, roundScale must be true, and groupIndex and scale are not supported.
  clampLimit must be a finite value; -1.0 indicates that Clamp is disabled, otherwise it must be greater than 0.
- When quantMode is 2 or 3, HIFP8 quantization output is supported, and dstType, blockSize, and roundScale do not take effect. The dimensions of input x are [T, D] or [B, S, D], which must meet the following specification constraints:

  | Specification item | Specification | Description |
  | :--- | :--- | :--- |
  | B | 1~31 | - |
  | S | 0~128K | - |
  | D/2 | 512, 768, 1024, 1536, 1792, 2048, 2560, 4096 | - |
  | dstTypeMax | 15, 56, 224, 32768 | - |

- The data type of yScale must match quantMode: FLOAT32 for Block FP8, FLOAT8_E8M0 for MX, and FLOAT32 for HIFP8.
- Element values in groupIndex must be greater than or equal to 0.

## Invocation Description

|Invocation method|Invocation sample|Description|
|:-------|:-------|:---|
|aclnn invocation|[test_aclnn_swiglu_group_quant](./examples/test_aclnn_swiglu_group_quant.cpp)|Calls the SwigluGroupQuant operator through the [aclnnSwigluGroupQuant](./docs/aclnnSwigluGroupQuant_en.md) interface.|
|aclnn V2 invocation|[test_aclnn_swiglu_group_quant_v2](./examples/test_aclnn_swiglu_group_quant_v2.cpp)|Calls quant_mode=5 of the SwigluGroupQuant operator through the [aclnnSwigluGroupQuantV2](./docs/aclnnSwigluGroupQuantV2.md) interface.|
|Graph-mode invocation|-|Calls the SwigluGroupQuant operator by constructing a graph through the [operator IR](./op_graph/swiglu_group_quant_proto.h).|
