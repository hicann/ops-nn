# aclnnSwigluGroupQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T15:38:28.121Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

### Description

The SwigluGroupQuant operator implements the fused computation of the SwiGLU activation function and group quantization. Four quantization modes are supported:

- **quant_mode=0**: Block Quant (FP8 block quantization, fixed group of 128 elements)
- **quant_mode=1**: MX Quant (FP8 MX quantization, fixed group of 32 elements)
- **quant_mode=2**: HiFp8 Static Quant (HiFp8 static quantization)
- **quant_mode=3**: HiFp8 Dynamic Quant (HiFp8 dynamic quantization)

### Formula

**Basic Computation Process**

```txt
Step 1: GroupIndex processing (optional) → Compute real_bs
Step 2: Input splitting (processes only the first real_bs rows)
Step 3: Clamp processing (optional, processes only the first real_bs rows)
Step 4: SwiGLU activation (only the first real_bs rows are processed)
Step 5: Weight weighting (optional, only the first real_bs rows are processed)
Step 6: Quantization computation (only the first real_bs rows are processed)
```

<details>
<summary><strong>Step 1: GroupIndex Processing (Optional)</strong></summary>

When `group_index` is provided, it is used to dynamically compute the quantity of tokens actually processed:

$$
\text{group\_sum} = \sum_{g=0}^{G-1} \text{group\_index}[g]
$$

$$
\text{real\_bs} = \min(\text{group\_sum}, N)
$$

Where:

- $G$ is the number of MoE expert groupings
- $N$ is the first dimension of the input tensor (preset batch size)
- All subsequent steps process only the first $\text{real\_bs}$ rows of data

**MoE Scenario Description**: In MoE inference, different experts may process different quantities of tokens. group_index allows dynamically adjusting the processing range to avoid processing empty data.

</details>

<details>
<summary><strong>Step 2: Input Splitting</strong></summary>

The input tensor $\mathbf{x} \in \mathbb{R}^{N \times D}$ is split into two parts along the last dimension:

$$
\mathbf{x}_0[n, d] = \mathbf{x}[n, d], \quad d \in [0, D/2)
$$

$$
\mathbf{x}_1[n, d] = \mathbf{x}[n, d + D/2], \quad d \in [0, D/2)
$$

</details>

<details>
<summary><strong>Step 3: Clamp Processing (Optional)</strong></summary>

When `clamp_limit > 0`, the input is clamped:

$$
\mathbf{x}_0'[n, d] = \min(\mathbf{x}_0[n, d], c)
$$

$$
\mathbf{x}_1'[n, d] = \min(\max(\mathbf{x}_1[n, d], -c), c)
$$

Where $c$ is `clamp_limit`.

**Purpose of Clamp**:

- $\mathbf{x}_0$ (gating branch) is limited to the positive range $[0, c]$ to prevent sigmoid gradient vanishing
- $\mathbf{x}_1$ (linear branch) is limited to the symmetric range $[-c, c]$ to prevent numerical overflow

</details>

<details>
<summary><strong>Step 4: SwiGLU Activation</strong></summary>

Definition of the SwiGLU activation function (element-wise computation):

$$
\mathbf{y}_{\text{swiglu}}[n, d] = \text{Swish}(\mathbf{x}_0'[n, d]) \cdot \mathbf{x}_1'[n, d]
$$

where the Swish function is:

$$
\text{Swish}(z) = z \cdot \sigma(z) = z \cdot \frac{1}{1 + e^{-z}}
$$

**Complete computation step breakdown**:

$$
\begin{aligned}
t_1[n, d] &= -\mathbf{x}_0'[n, d] \quad \text{(neg)} \\
t_2[n, d] &= e^{t_1[n, d]} = e^{-\mathbf{x}_0'[n, d]} \quad \text{(exp)} \\
t_3[n, d] &= t_2[n, d] + 1 = 1 + e^{-\mathbf{x}_0'[n, d]} \quad \text{(add)} \\
t_4[n, d] &= \frac{\mathbf{x}_0'[n, d]}{t_3[n, d]} = \text{Swish}(\mathbf{x}_0'[n, d]) \quad \text{(div)} \\
\mathbf{y}_{\text{swiglu}}[n, d] &= t_4[n, d] \cdot \mathbf{x}_1'[n, d] \quad \text{(mul)}
\end{aligned}
$$

**yOrigin output**:
When outputOrigin is set to True, `yOrigin` outputs the SwiGLU result $\mathbf{y}_{\text{swiglu}}$.

</details>

<details>
<summary><strong>Step 5: Weight Application (Optional)</strong></summary>

When `weight` is provided, the SwiGLU output is weighted:

$$
\mathbf{y}_{\text{weighted}}[n, d] = \mathbf{y}_{\text{swiglu}}[n, d] \cdot w[n]
$$

where $w[n]$ is the weight value of the $n$-th token.

**MoE use case**: The weight comes from the softmax output of the expert router and indicates the weight of the token for the current expert.

</details>

<details>
<summary><strong>Step 6: Quantization Computation</strong></summary>

<details>
<summary><strong>quant_mode=0 (Block Quant)</strong></summary>

**Group division**: Divide the output along the last dimension into groups of 128 elements:

$$
\mathbf{y} = [\mathbf{g}_0, \mathbf{g}_1, \ldots, \mathbf{g}_K], \quad K = \lceil D/2 / 128 \rceil
$$

Each group $\mathbf{g}_i \in \mathbb{R}^{N \times 128}$.

**Non-finite value masking and absolute value computation**:

$$
\begin{aligned}
\mathbf{z}[n, j] &= \mathbf{y}_{\text{weighted}}[n, j] \cdot 0 \quad \text{( generate a zero tensor)} \\
\mathbf{m}_{\text{finite}}[n, j] &= (\mathbf{z}[n, j] = \mathbf{z}[n, j]) \\
\mathbf{y}_{\text{abs}}[n, j] &=
\begin{cases}
|\mathbf{y}_{\text{weighted}}[n, j]|, & \mathbf{m}_{\text{finite}}[n, j] \\
0, & \text{otherwise}
\end{cases}
\end{aligned}
$$

**Masking principle**: The property of NaN is `NaN != NaN`; meanwhile, `Inf * 0` also produces NaN. Therefore, this step masks NaN and Inf when computing amax.

**Scale computation**:

For the $i$-th group (containing 128 consecutive elements):

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

**Scale Output and InvScale Computation**:

When `round_scale=false`:

$$
s_i = s_i^{\text{raw}}, \quad \text{InvScale}_i = \frac{M_{\text{fp8}}}{\hat{a}_i} = \frac{1}{s_i}
$$

When `round_scale=true`, the scale is rounded up to the nearest power of 2:

$$
e_i = \lceil \log_2(s_i^{\text{raw}}) \rceil
$$

$$
s_i = 2^{e_i}, \quad \text{InvScale}_i = 2^{-e_i}
$$

where $s_i$ is written to the FLOAT32 scale output.

**Quantization Computation**:

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

Where `cast_fp8_rint` indicates the type conversion from FP32 to FP8, using the **RINT (round to nearest)** mode.

</details>

<details>
<summary><strong>quant_mode=1 (MX Quant)</strong></summary>

**MX Quantization Principle**: Uses the combination of **E8M0 Scale** + **FP8 Data**.

**Grouping**: Every **32 elements** form a group:

$$
\mathbf{y} = [\mathbf{g}_0, \mathbf{g}_1, \ldots, \mathbf{g}_K], \quad \mathbf{g}_i \in \mathbb{R}^{32}
$$

**Amax Computation**:

$$
a_i = \max_{j=0}^{31} |\mathbf{g}_i[j]|
$$

$$
\hat{a}_i = \max(a_i, 10^{-4})
$$

**Raw Scale Computation**:

$$
s_i^{\text{raw}} = \frac{\hat{a}_i}{M_{\text{fp8}}}
$$

where $M_{\text{fp8}}$ takes the following values:

- FP8 E4M3FN: $M_{\text{fp8}} = 448.0$
- FP8 E5M2: $M_{\text{fp8}} = 57344.0$

quant_mode=1 only supports `round_scale=true`, which rounds the original scale up to a power of 2:

$$
e_i = \lceil \log_2(s_i^{\text{raw}}) \rceil
$$

This is equivalent to the computation based on the FP32 bit pattern:

$$
e_i = E(s_i^{\text{raw}}) - 127 + \mathbf{1}_{\text{mantissa}(s_i^{\text{raw}}) \ne 0}
$$

**E8M0 Scale Encoding**:

$$
s_i^{\text{e8m0}} = e_i + 127
$$

where $s_i^{\text{e8m0}}$ is written to the scale output of the FLOAT8_E8M0 type, indicating that the actual scale value is $2^{e_i}$.

**InvScale Computation**:

$$
\text{InvScale}_i = 2^{-e_i}
$$

**Quantization Computation**:

$$
\mathbf{y}_{\text{quant}}[j] = \text{cast\_fp8\_rint}\left(\mathbf{y}_{\text{weighted}}[j] \cdot \text{InvScale}_i\right), \quad j \in \text{group } i
$$

</details>

<details>
<summary><strong>quant_mode=2 (HiFp8 Static Quant)</strong></summary>

**Static Quantization Description**: The weighted SwiGLU output is scaled and quantized using the pre-provided `invScale`.

**Case 1: No GroupIndex** (groupIndex is empty):

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\mathbf{y}_{\text{weighted}}[n, d] \cdot \text{invScale}[0]\right), \quad n \in [0, N), \quad d \in [0, D/2)

$$
where `hif8_cast` indicates the HiFloat8 data type conversion function.

**Case 2: With GroupIndex** (groupIndex is not empty):

Let $G$ be the number of MoE expert groups, and $\text{groupIndex}[g]$ indicates the quantity of tokens processed by the $g$-th expert.

Compute the start and end indices of each group:

$$
\text{start}^{(0)} = 0, \quad \text{end}^{(g)} = \sum_{k=0}^{g} \text{groupIndex}[k], \quad \text{start}^{(g)} = \text{end}^{(g-1)}
$$

For the $g$-th group, quantization is performed using the corresponding scale factor $\text{invScale}[g]$:

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\mathbf{y}_{\text{weighted}}[n, d] \cdot \text{invScale}[g]\right), \quad n \in [\text{start}^{(g)}, \text{end}^{(g)}), \quad d \in [0, D/2)
$$

**MoE Scenario Description**: In MoE inference, different experts process different quantities of tokens. groupIndex identifies the token range processed by each expert, and invScale is the pre-computed static scale factor for each expert.

</details>

<details>
<summary><strong>quant_mode=3 (HiFp8 Dynamic Quant)</strong></summary>

**Dynamic Quantization Description**: The scale factor is dynamically computed based on the weighted SwiGLU output for quantization.

**Case 1: No GroupIndex** (groupIndex is empty):

The global maximum absolute value is computed:

$$
a_{\max} = \max\left(\max_{n \in [0, N), d \in [0, D/2)} |\mathbf{y}_{\text{weighted}}[n, d]|, \epsilon\right)
$$

Where $\epsilon$ is a numerical stability constant.

Compute the scaling factor:

$$
s = \frac{a_{\max}}{M_{\text{hif8}}}
$$

Where $M_{\text{hif8}}$ is `dstTypeMax`, indicating the maximum finite value of the HiFloat8 data type.

Quantization computation:

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\frac{\mathbf{y}_{\text{weighted}}[n, d]}{s}\right), \quad n \in [0, N), \quad d \in [0, D/2)
$$

Where `hif8_cast` is the HiFloat8 type conversion function.

**Case 2: With GroupIndex** (groupIndex is not empty):

Let $G$ be the number of MoE expert groups. $\text{groupIndex}[g]$ indicates the quantity of tokens processed by the $g$-th expert.

Compute the start and end indices of each group:

$$
\text{start}^{(0)} = 0, \quad \text{end}^{(g)} = \sum_{k=0}^{g} \text{groupIndex}[k], \quad \text{start}^{(g)} = \text{end}^{(g-1)}
$$

For the $g$-th group, extract the corresponding data:

$$
\mathbf{y}^{(g)} = \mathbf{y}_{\text{weighted}}[\text{start}^{(g)}:\text{end}^{(g)}, :]
$$

Compute the maximum absolute value of this group:

$$
a_{\max}^{(g)} = \max\left(\max_{n \in [\text{start}^{(g)}, \text{end}^{(g)}), d \in [0, D/2)} |\mathbf{y}^{(g)}[n, d]|, \epsilon\right)
$$

Compute the scaling factor for the group:

$$
s^{(g)} = \frac{a_{\max}^{(g)}}{M_{\text{hif8}}}
$$

Quantize the group:

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\frac{\mathbf{y}_{\text{weighted}}[n, d]}{s^{(g)}}\right), \quad n \in [\text{start}^{(g)}, \text{end}^{(g)}), \quad d \in [0, D/2)
$$

**MoE Scenario Description**: In MoE inference, different experts process different quantities of tokens. groupIndex indicates the token range processed by each expert. Each group independently computes the scaling factor to adapt to different data distributions.

</details>

</details>

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnSwigluGroupQuantGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the computation process of the operator, and then call the "aclnnSwigluGroupQuant" API to execute the computation.

```Cpp
aclnnStatus aclnnSwigluGroupQuantGetWorkspaceSize(
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
  const aclTensor *yOut,
  const aclTensor *yScaleOut,
  const aclTensor *yOriginOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSwigluGroupQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnSwigluGroupQuantGetWorkspaceSize

- **Parameters:**

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
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>SwiGLU input.</td>
      <td><ul><li>The shape is [...,D].</li><li>D must be divisible by 256.</li><li>The number of dimensions must be 2 to 8, or 2 to 7 when quantMode is 1.</li><li>Supports only FLOAT16 and BFLOAT16 when quantMode is 0 or 1. Supports FLOAT, FLOAT16, and BFLOAT16 when quantMode is 2 or 3.</li><li>Empty tensors are supported when quantMode is 0 or 1, but not supported when quantMode is 2 or 3.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional (aclTensor*)</td>
      <td>Input (Optional)</td>
      <td>MoE weight tensor used for weighted computation of the SwiGLU output.</td>
      <td><ul><li>Optional Parameter. When quantMode is 0 or 1, empty tensors are supported; when quantMode is 2 or 3, empty tensors are not supported.</li><li>When not empty, the data type is FLOAT32, and the number of elements must equal the product of the number of elements of x in all dimensions except the last.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input (Optional)</td>
      <td>Group token quantity in count mode.</td>
      <td><ul><li>Optional Parameter. When quantMode is 0 or 1, this parameter cannot be an empty tensor alone; when quantMode is 2 or 3, empty tensors are not supported.</li><li>When not empty, the data type is INT64, and the shape is [G].</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scaleOptional (aclTensor*)</td>
      <td>Input (Optional)</td>
      <td>invScale tensor of the static quantization input when quantMode=2.</td>
      <td><ul><li>Optional parameter, used only when quant_mode is 2.</li><li>shape=[G] if groupIndex exists, and shape=[1] otherwise.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Target quantization type.</td>
      <td><ul><li>This parameter takes effect only when quantMode is 0 or 1.</li><li>Values 35 and 36 are supported, indicating FLOAT8_E5M2 and FLOAT8_E4M3FN respectively.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode (int64_t)</td>
      <td>Input</td>
      <td>Quantization mode.</td>
      <td><ul><li>Supported values: 0, 1, 2, and 3.</li><li>0 indicates Block FP8 mode.</li><li>1 indicates MX mode.</li><li>2 indicates HIFP8 static quantization mode.</li><li>3 indicates HIFP8 dynamic quantization mode.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize (int64_t)</td>
      <td>Input</td>
      <td>Quantization block size.</td>
      <td><ul><li>0 indicates the default block size of the current quantization mode.</li><li>When quantMode is 0, 0 or 128 is supported.</li><li>When quantMode is 1, 0 or 32 is supported.</li><li>When quantMode is 2 or 3, this parameter does not take effect and defaults to 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundScale (bool)</td>
      <td>Input</td>
      <td>Specifies whether to round scale to a power of 2.</td>
      <td><ul><li>When quantMode is 1, roundScale must be true.</li><li>When quantMode is 2 or 3, this parameter does not take effect.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clampLimit (double)</td>
      <td>Input</td>
      <td>Clamp threshold before SwiGLU computation.</td>
      <td><ul><li>-1.0 indicates that clamp is disabled.</li><li>If clamp is enabled, clampLimit must be greater than 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax (double)</td>
      <td>Input</td>
      <td>Maximum finite value of the target quantization type.</td>
      <td><ul><li>This parameter takes effect only when quantMode is 3.</li><li>The default value is 15.0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputOrigin (bool)</td>
      <td>Input</td>
      <td>Indicates whether to output the SwiGLU result before weight weighting.</td>
      <td><ul><li>true indicates that yOrigin is output. If false, the yOrigin output is invalid.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Quantization output.</td>
      <td><ul><li>When quantMode is 0 or 1, the data type must be consistent with dstType; when quantMode is 2 or 3, the data type defaults to HIFLOAT8.</li><li>All shapes are [...,D/2].</li><li>Empty tensors are supported when quantMode is 0 or 1; empty tensors are not supported when quantMode is 2 or 3.</li></ul></td>
      <td>HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>yScaleOut(aclTensor*)</td>
      <td>Output</td>
      <td>Quantization scale output.</td>
      <td><ul><li>When quantMode is 0, the shape is [...,ceil((D/2)/128)] and the data type is FLOAT32.</li><li>When quantMode is 1, the shape is [...,ceil(ceil((D/2)/32)/2),2] and the data type is FLOAT8_E8M0.</li><li>When quantMode is 3, the shape is [1] without groupIndex and [G] with groupIndex, and the data type is FLOAT32.</li><li>Empty tensors are supported when quantMode is 0 or 1; empty tensors are not supported when quantMode is 2 or 3.</li></ul></td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>yOriginOut (aclTensor*)</td>
      <td>Output</td>
      <td>SwiGLU result before weight weighting.</td>
      <td><ul><li>The shape is [...,D/2].</li><li>The data type must be the same as that of x.</li><li>Null pointers are not supported.</li><li>Empty tensors are supported when quantMode is 0 or 1; empty tensors are not supported when quantMode is 2 or 3.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return value**

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x, yOut, yScaleOut, yOriginOut, workspaceSize, or executor is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of the input or output is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of the input or output does not meet the constraints.</td>
    </tr>
    <tr>
      <td>dstType, quantMode, blockSize, roundScale, or clampLimit does not match the currently supported values.</td>
    </tr>
    <tr>
      <td>The data type combination of dstType, quantMode, blockSize, roundScale, and yScaleOut does not match.</td>
    </tr>
    <tr>
      <td>weightOptional or groupIndexOptional does not meet the optional input constraints.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The shape information of multiple input tensors is mismatched, or the input attributes are out of the value range (see Parameters).</td>
    </tr>
  </tbody></table>

## aclnnSwigluGroupQuant

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by the first-phase API aclnnSwigluGroupQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody></table>

- **Return value**

 aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnSwigluGroupQuant defaults to a deterministic implementation.
- When quantMode is 0, only FP8 output is supported, and blockSize can be 0 or 128.
- When quantMode is 1, FP8 output is supported, blockSize can be 0 or 32, and roundScale must be true.
- When quantMode is 2 or 3, HIFP8 quantization output is supported, and dstType, blockSize, and roundScale do not take effect. The dimensions of input x are [T, D] or [B, S, D], which must meet the following specification constraints:

  | Specification Item | Specification | Specification Description |
  | :--- | :--- | :--- |
  | B | 1~31 | - |
  | S | 0~128K | - |
  | D/2 | 512, 768, 1024, 1536, 1792, 2048, 2560, 4096 | - |
  | dstTypeMax | 15, 56, 224, 32768 | - |

- The data type of yScale must match quantMode: when quantMode is 0 or 3, the data type is FLOAT32; when quantMode is 1, the data type is FLOAT8_E8M0.
- Each element in groupIndex must be greater than or equal to 0.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_group_quant.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

bool CheckHardwareSupport()
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        LOG_PRINT("Warning: Cannot get SOC name, skip hardware check\n");
        return true;
    }

    LOG_PRINT("Current SOC: %s\n", socName);
    if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
        return true;
    }

    LOG_PRINT("Warning: SwigluGroupQuant only supports Ascend950, current SOC '%s' is not supported. Skip test.\n",
              socName);
    return false;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
}

struct AclTensorResource {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;
};

void DestroyAclTensorResource(AclTensorResource& resource)
{
    if (resource.tensor != nullptr) {
        aclDestroyTensor(resource.tensor);
        resource.tensor = nullptr;
    }
    if (resource.deviceAddr != nullptr) {
        aclrtFree(resource.deviceAddr);
        resource.deviceAddr = nullptr;
    }
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

struct SwigluGroupQuantCase {
    const char* name;
    int64_t quantMode;
    int64_t blockSize;
    bool roundScale;
    std::vector<int64_t> yScaleShape;
    aclDataType yScaleDataType;
    bool yScaleIsE8M0;
    int64_t dstType;
    double dstTypeMax;
    aclDataType yDataType;
    int64_t yElementSize;
    bool outputOrigin;
    bool hasScaleInput;
    std::vector<int64_t> scaleShape;
};

int RunSwigluGroupQuantCase(const SwigluGroupQuantCase& testCase, aclrtStream stream)
{
    std::vector<int64_t> xShape = {2, 256};
    std::vector<int64_t> yShape = {2, 128};
    std::vector<int64_t> yOriginShape = {2, 128};

    std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0);
    for (size_t i = 0; i < xHostData.size(); ++i) {
        xHostData[i] = static_cast<uint16_t>(i % 23);
    }
    std::vector<uint8_t> yHostData(GetShapeSize(yShape) * testCase.yElementSize, 0);
    std::vector<uint8_t> yScaleE8M0HostData(GetShapeSize(testCase.yScaleShape), 0);
    std::vector<float> yScaleFp32HostData(GetShapeSize(testCase.yScaleShape), 0.0f);
    std::vector<uint16_t> yOriginHostData(GetShapeSize(yOriginShape), 0);
    std::vector<float> scaleHostData;
    if (testCase.hasScaleInput) {
        scaleHostData.resize(GetShapeSize(testCase.scaleShape), 1.0f);
    }

    AclTensorResource xResource;
    AclTensorResource yResource;
    AclTensorResource yScaleResource;
    AclTensorResource yOriginResource;
    AclTensorResource scaleResource;

    auto ret = CreateAclTensor(xHostData, xShape, &xResource.deviceAddr, ACL_FLOAT16, &xResource.tensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yResource.deviceAddr, testCase.yDataType, &yResource.tensor);
    CHECK_RET(ret == ACL_SUCCESS, DestroyAclTensorResource(xResource); return ret);
    if (testCase.yScaleIsE8M0) {
        ret = CreateAclTensor(yScaleE8M0HostData, testCase.yScaleShape, &yScaleResource.deviceAddr,
                              testCase.yScaleDataType, &yScaleResource.tensor);
    } else {
        ret = CreateAclTensor(yScaleFp32HostData, testCase.yScaleShape, &yScaleResource.deviceAddr,
                              testCase.yScaleDataType, &yScaleResource.tensor);
    }
    CHECK_RET(ret == ACL_SUCCESS, DestroyAclTensorResource(xResource); DestroyAclTensorResource(yResource);
              DestroyAclTensorResource(yScaleResource); return ret);
    ret = CreateAclTensor(yOriginHostData, yOriginShape, &yOriginResource.deviceAddr, ACL_FLOAT16,
                          &yOriginResource.tensor);
    CHECK_RET(ret == ACL_SUCCESS, DestroyAclTensorResource(xResource); DestroyAclTensorResource(yResource);
              DestroyAclTensorResource(yScaleResource); DestroyAclTensorResource(yOriginResource); return ret);
    if (testCase.hasScaleInput) {
        ret = CreateAclTensor(scaleHostData, testCase.scaleShape, &scaleResource.deviceAddr,
                              ACL_FLOAT, &scaleResource.tensor);
        CHECK_RET(ret == ACL_SUCCESS, DestroyAclTensorResource(xResource); DestroyAclTensorResource(yResource);
                  DestroyAclTensorResource(yScaleResource); DestroyAclTensorResource(yOriginResource);
                  DestroyAclTensorResource(scaleResource); return ret);
    }

    double clampLimit = -1.0;
    bool outputOrigin = testCase.outputOrigin;
    const aclTensor* scaleTensor = testCase.hasScaleInput ? scaleResource.tensor : nullptr;

    LOG_PRINT("Run %s: quant_mode=%ld\n", testCase.name, testCase.quantMode);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnSwigluGroupQuantGetWorkspaceSize(xResource.tensor, nullptr, nullptr, scaleTensor, testCase.dstType,
                                                testCase.quantMode, testCase.blockSize, testCase.roundScale, clampLimit,
                                                testCase.dstTypeMax, outputOrigin, yResource.tensor,
                                                yScaleResource.tensor, yOriginResource.tensor, &workspaceSize,
                                                &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              DestroyAclTensorResource(xResource); DestroyAclTensorResource(yResource);
              DestroyAclTensorResource(yScaleResource); DestroyAclTensorResource(yOriginResource);
              DestroyAclTensorResource(scaleResource); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  DestroyAclTensorResource(xResource); DestroyAclTensorResource(yResource);
                  DestroyAclTensorResource(yScaleResource); DestroyAclTensorResource(yOriginResource);
                  DestroyAclTensorResource(scaleResource); return ret);
    }

    ret = aclnnSwigluGroupQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuant failed. ERROR: %d\n", ret);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } DestroyAclTensorResource(xResource);
        DestroyAclTensorResource(yResource); DestroyAclTensorResource(yScaleResource);
        DestroyAclTensorResource(yOriginResource); DestroyAclTensorResource(scaleResource); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } DestroyAclTensorResource(xResource);
        DestroyAclTensorResource(yResource); DestroyAclTensorResource(yScaleResource);
        DestroyAclTensorResource(yOriginResource); DestroyAclTensorResource(scaleResource); return ret);

    std::vector<uint8_t> resultData(GetShapeSize(yShape) * testCase.yElementSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yResource.deviceAddr,
                      resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } DestroyAclTensorResource(xResource);
        DestroyAclTensorResource(yResource); DestroyAclTensorResource(yScaleResource);
        DestroyAclTensorResource(yOriginResource); DestroyAclTensorResource(scaleResource); return ret);
    LOG_PRINT("%s result[0] is: %d\n", testCase.name, resultData[0]);

    DestroyAclTensorResource(xResource);
    DestroyAclTensorResource(yResource);
    DestroyAclTensorResource(yScaleResource);
    DestroyAclTensorResource(yOriginResource);
    DestroyAclTensorResource(scaleResource);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    if (!CheckHardwareSupport()) {
        LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
        Finalize(deviceId, stream);
        return ACL_SUCCESS;
    }

    std::vector<SwigluGroupQuantCase> testCases = {
        {"block_fp8", 0, 0, false, {2, 1}, ACL_FLOAT, false, 36, 448.0, ACL_FLOAT8_E4M3FN, 1, false, false, {}},
        {"block_fp8_y_origin", 0, 0, false, {2, 1}, ACL_FLOAT, false, 36, 448.0, ACL_FLOAT8_E4M3FN, 1, true, false, {}},
        {"mx_fp8", 1, 0, true, {2, 2, 2}, ACL_FLOAT8_E8M0, true, 36, 448.0, ACL_FLOAT8_E4M3FN, 1, false, false, {}},
        {"mx_fp8_y_origin", 1, 0, true, {2, 2, 2}, ACL_FLOAT8_E8M0, true, 36, 448.0, ACL_FLOAT8_E4M3FN, 1, true,
         false, {}},
        {"hifp8_static", 2, 0, false, {1}, ACL_FLOAT, false, 27, 448.0, ACL_HIFLOAT8, 1, false, true, {1}},
        {"hifp8_dynamic", 3, 0, false, {1}, ACL_FLOAT, false, 27, 15.0, ACL_HIFLOAT8, 1, false, false, {}},
    };

    for (const auto& testCase : testCases) {
        ret = RunSwigluGroupQuantCase(testCase, stream);
        CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);
    }

    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
```
