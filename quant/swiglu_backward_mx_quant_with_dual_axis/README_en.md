# SwigluBackwardMxQuantWithDualAxis

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor series products</term> | × |
| <term>Kirin 9030 processor series products</term> | × |

## Function

- Description: Fused operator that implements the combined computation of SwiGLU activation function backward gradient computation and dual-axis dynamic block quantization. The operator first computes the SwiGLU backward gradient for the input x and the gradient yGrad, then performs block-based dynamic quantization on the gradient results along the -1 axis and the -2 axis respectively, and outputs low-precision FP8 tensors and the corresponding scaling factors.

- Formula:

  **Stage 1: SwiGLU backward gradient computation**

  - When `activateLeft=true`:

  $$
  A_i = x_i \text{的前半部分}, B_i = x_i \text{的后半部分}
  $$

  - When `activateLeft=false`:

  $$
  A_i = x_i \text{的后半部分}, B_i = x_i \text{的前半部分}
  $$

  - SwiGLU forward computation:

  $$
  swigluOut_i = Swish(A_i) * B_i = \frac{A_i}{1 + e^{-A_i}} * B_i
  $$

  - SwiGLU backward computation:

  $$
  grad\_A = yGrad * B * sigmoid(A) * (1 + A - (A * sigmoid(A))) \\
  grad\_B = yGrad * A * sigmoid(A) \\
  grad\_x = concat(grad\_A, grad\_B)
  $$

  **Stage 2: dual-axis dynamic block quantization**

- **-1 axis quantization (column direction)**: Groups the SwiGLU result into groups of 32 values along the -1 axis; a group of 32 values $\{\{V_i\}_{i=1}^{32}\}$ is quantized to $\{mxscale1, \{P_i\}_{i=1}^{32}\}$

  $$
  shared\_exp = floor(log_2(max_i(|V_i|))) - emax
  $$

  $$
  mxscale1 = 2^{shared\_exp}
  $$

  $$
  P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32
  $$
- **-2 axis quantization (row direction)**: Groups the SwiGLU result into groups of 32 values along the -2 axis; a group of 32 values $\{\{V_j\}_{j=1}^{32}\}$ is quantized to $\{mxscale2, \{P_j\}_{j=1}^{32}\}$

  $$
  shared\_exp = floor(log_2(max_j(|V_j|))) - emax
  $$

  $$
  mxscale2 = 2^{shared\_exp}
  $$

  $$
  P_j = cast\_to\_dst\_type(V_j/mxscale2, round\_mode), \space j\space from\space 1\space to\space 32
  $$
- The $P_{i}$ values quantized along the -1 axis form the output y1Out at the positions of the corresponding $V_{i}$, and mxscale1 forms the output mxscale1Out according to the groups on the -1 axis dimension. The $P_{j}$ values quantized along the -2 axis form the output y2Out at the positions of the corresponding $V_{j}$, and mxscale2 forms the output mxscale2Out according to the groups on the -2 axis dimension.
- emax: the exponent of the largest normal number of the corresponding data type.

  |   DataType   | emax |
  | :-----------: | :--: |
  |  FLOAT4_E2M1  |  2  |
  |  FLOAT4_E1M2  |  0  |
  | FLOAT8_E4M3FN |  8  |
  |  FLOAT8_E5M2  |  15  |

## Parameters

<table><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Usage instructions</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input tensor, which is x in the formulas.</td>
      <td><ul><li>The shape is [M, 2N], and the last dimension must be an even number.</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_grad</td>
      <td>Input</td>
      <td>Backward gradient input, with shape [M, N], that is, the gradient of the forward SwiGLU output.</td>
      <td><ul><li>The shape is [M, N], where N = last dimension of x / 2.</li><li>The data type must be the same as that of x.</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>Optional Input</td>
      <td>Group index, used to control group quantization boundaries.</td>
      <td><ul><li>The shape is [G]; it uses the cumsum mode and represents the cumulative row count of each group. Passing a null pointer means no grouping.</li><li>Passing an empty tensor is currently not supported.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>Optional Attribute</td>
      <td>Selection of the SwiGLU activation side.</td>
      <td><ul><li>True means that the left half is hidden and the right half is gate.</li><li>False means that the right half is hidden and the left half is gate.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Optional Attribute</td>
      <td>Represents the data conversion mode, corresponding to round_mode in the formulas.</td>
      <td><ul><li>Only dstDtype 36 is supported; when the data types of the outputs y1Out and y2Out are FLOAT8_E4M3FN, only {"rint"} is supported.</li><li>When a null pointer is passed in, the "rint" mode is used.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>Optional Attribute</td>
      <td>Represents the computation methods of mxscale1Out and mxscale2Out.</td>
      <td><ul><li>Value range: {1}, which represents the cuBLAS implementation.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_dtype</td>
      <td>Optional Attribute</td>
      <td>Specifies the types of y1Out and y2Out after data conversion.</td>
      <td><ul><li>The input range is {36}, and the data types of the outputs y1Out and y2Out correspond to {36: FLOAT8_E4M3FN}.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_dtype_value</td>
      <td>Optional Attribute</td>
      <td>Reserved parameter.</td>
      <td><ul><li>Only 0 is supported for maxDtypeValue.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1_out</td>
      <td>Output</td>
      <td>Represents the result corresponding to the SwiGLU result after quantization along the -1 axis, corresponding to <i>P<sub>i</sub></i> in the formulas.</td>
      <td><ul><li>The shape is [M, 2N].</li></ul></td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale1_out</td>
      <td>Output</td>
      <td>Represents the quantization scale corresponding to each group on the -1 axis, corresponding to mxscale1 in the formulas.</td>
      <td><ul><li>The shape is [M, (ceil(N/32)+2-1)/2, 2]; even padding is required, and the pad fill value is 0.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2_out</td>
      <td>Output</td>
      <td>Represents the result corresponding to the SwiGLU result after quantization along the -2 axis, corresponding to <i>P<sub>j</sub></i> in the formulas.</td>
      <td><ul><li>The shape is [M, 2N].</li></ul></td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale2_out</td>
      <td>Output</td>
      <td>Represents the quantization scale corresponding to each group on the -2 axis, corresponding to mxscale2 in the formulas.</td>
      <td><ul><li>When groupIndexOptional exists, the shape is [floor(M/64)+G, N, 2].</li><li>Even padding is required, and the pad fill value is 0.</li><li>The mxscale2Out output requires interleaving processing for every two rows of data.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The last dimension of the input x must be divisible by 64.
- The FP8 output type (FLOAT8_E4M3FN) supports only the "rint" rounding mode.
- groupIndexOptional uses the cumsum mode; each value represents the cumulative row count of the corresponding group. Each element value of groupIndexOptional must be greater than 0, and the last element value must equal M.

## Invocation Description

| Invocation method | Invocation sample | Description |
| :------- | :------- | :--- |
| aclnn invocation | [test_aclnn_swiglu_backward_mx_quant_with_dual_axis](./examples/arch35/test_aclnn_swiglu_backward_mx_quant_with_dual_axis.cpp) | Invokes the SwigluBackwardMxQuantWithDualAxis operator through the [aclnnSwigluBackwardMxQuantWithDualAxis](./docs/aclnnSwigluBackwardMxQuantWithDualAxis_en.md) interface. |
