# SwigluMxQuantWithDualAxis

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

- Operator function: Fused operator that implements the combined computation of the SwiGLU activation function and dual-axis dynamic block quantization. It first computes the SwiGLU activation function on the input, then performs block-based dynamic quantization on the result along both the -1 axis and the -2 axis, and outputs low-precision FP4/FP8 tensors and the corresponding scale factors.

- Formula:

  **Stage 1: SwiGLU activation function**

  <p style="text-align: center">
  gate, hidden = split(x, axis=-1)
  </p>
  <p style="text-align: center">
  swish = sigmoid(gate) * gate
  </p>
  <p style="text-align: center">
  act = swish * hidden
  </p>

  When `activate_left=True`, the left half is hidden and the right half is gate; when `activate_left=False`, the right half is hidden and the left half is gate.

  **Stage 2: Dual-axis dynamic block quantization**

  - **-1 axis quantization (column direction)**: Group the SwiGLU result along the -1 axis by 32 values, and quantize a group of 32 values $\{\{V_i\}_{i=1}^{32}\}$ into $\{mxscale1, \{P_i\}_{i=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale1 = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32
    $$

  - **-2 axis quantization (row direction)**: Group the SwiGLU result along the -2 axis by 32 values, and quantize a group of 32 values $\{\{V_j\}_{j=1}^{32}\}$ into $\{mxscale2, \{P_j\}_{j=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_j(|V_j|))) - emax
    $$

    $$
    mxscale2 = 2^{shared\_exp}
    $$

    $$
    P_j = cast\_to\_dst\_type(V_j/mxscale2, round\_mode), \space j\space from\space 1\space to\space 32
    $$

  - emax: Exponent bits of the maximum regular number of the corresponding data type.

    |   DataType    | emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

## Parameters

<table><colgroup>
  <col style="width: 120px">
  <col style="width: 100px">
  <col style="width: 380px">
  <col style="width: 280px">
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
      <td>Input tensor. It must be 2D, and the last dimension must be an even number (shape [M, 2N])</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>Optional input</td>
      <td>Group index, used to control the grouped quantization boundary. Shape [G], in cumsum mode, representing the cumulative number of rows of each group. Passing a null pointer indicates no grouping.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>Attribute</td>
      <td>Selection of the SwiGLU activation side. True indicates that the left half is hidden and the right half is gate; False indicates that the right half is hidden and the left half is gate. Default value: True.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Attribute</td>
      <td>Rounding mode, used for type conversion during quantization. <br>When dst_type is FLOAT8_E5M2/FLOAT8_E4M3FN, only "rint" is supported; when dst_type is FLOAT4_E2M1/FLOAT4_E1M2, "rint", "floor", and "round" are supported. Default value: "rint".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>Attribute</td>
      <td>Scale algorithm: value 1 indicates the cuBLAS algorithm; value 0 indicates the OCP algorithm. Default value: 0. <br>When dst_type is FLOAT4_E2M1/FLOAT4_E1M2, only scale_alg=0 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Attribute</td>
      <td>Target quantization type: 35=FLOAT8_E5M2, 36=FLOAT8_E4M3FN, 40=FLOAT4_E2M1, 41=FLOAT4_E1M2. Default value: 35.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_dtype_value</td>
      <td>Attribute</td>
      <td>Reserved parameter. It takes effect when scale_alg=2 and dst_type=FLOAT4_E1M2. <br>Currently only the value 0.0 is supported.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>Output tensor after quantization along the -1 axis, with shape [M, N] (half of the SwiGLU output).</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mx_scale1</td>
      <td>Output</td>
      <td>Scale factor corresponding to each quantization block along the -1 axis. Shape [M, (ceil(N/32)+2-1)/2, 2]; even padding is required, and the padding value is 0.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Output tensor after quantization along the -2 axis, with shape [M, N].</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mx_scale2</td>
      <td>Output</td>
      <td>Scale factor corresponding to each quantization block along the -2 axis. When groupIndex exists, the shape is [floor(M/64) + G, N, 2]; when groupIndex does not exist, the shape is [ceil(M/64), N, 2]. Even padding is required, and the padding value is 0. The output needs to be interleaved for every two rows of data.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Input x must be a 2D tensor, and the last dimension must be divisible by 2 (shape [M, 2N]).
- FP8 output types (FLOAT8_E5M2/FLOAT8_E4M3FN) support only the "rint" rounding mode; FP4 output types (FLOAT4_E2M1/FLOAT4_E1M2) support the "rint", "floor", and "round" rounding modes.
- When dst_type is FLOAT4_E2M1/FLOAT4_E1M2, only scale_alg=0 (OCP implementation) is supported.
- When group_index exists, cumsum mode is used. Each value represents the cumulative number of rows of the corresponding group. Each element value of group_index must be greater than 0, and the last element value must equal M.

## Invocation Description

| Invocation method | Invocation sample | Description |
| :------- | :------- | :--- |
| aclnn invocation | [test_aclnn_swiglu_mx_quant_with_dual_axis](./examples/arch35/test_aclnn_swiglu_mx_quant_with_dual_axis.cpp) | Calls the SwigluMxQuantWithDualAxis operator through the [aclnnSwigluMxQuantWithDualAxis](./docs/aclnnSwigluMxQuantWithDualAxis_en.md) interface. |
| Graph-mode invocation | - | Calls the SwigluMxQuantWithDualAxis operator by building a graph with the operator IR. |
