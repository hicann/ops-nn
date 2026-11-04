# GroupedDynamicMxQuantWithDualAxis

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: According to the grouping described by group_index, the operator performs dynamic MX quantization on the 2D input x in both the last-dimension and second-to-last-dimension directions at one time, and outputs the FP8 quantization results y1 and y2 in the two directions and the corresponding FLOAT8_E8M0 scale factors y1_scale and y2_scale respectively.
- Computation description: 32 elements form one MX block, and each block shares one scale of the FLOAT8_E8M0 type. The scale and quantization result of each block are computed independently for the -1 axis and the -2 axis.

- Formula:

  For the input data in one MX block:

  $$
  D^{b}=[d^{i}]_{i=1}^{k},\quad k=32
  $$

  Compute the maximum absolute value of the elements in the block:

  $$
  Amax(D^{b})=\max_i(|d^{i}|)
  $$

  When scale_alg=1, compute the FP32 scale based on the maximum representable value of the target FP8 type:

  $$
  S_{fp32}^{b}=\frac{Amax(D^{b})}{Amax(dst_dtype)}
  $$

  Extract the unbiased exponent $E_{int}^{b}$ and the mantissa $M_{fixp}^{b}$ from the block scale factor $S_{fp32}^{b}$.

  To ensure that no overflow occurs during quantization, round the exponent up, and keep it within the range representable by FP8:

  $$
  E_{int}^{b} = \begin{cases}
    E_{int}^{b} + 1, & \text{if } S_{fp32}^{b} \text{ is a normal number, } E_{int}^{b} < 254 \text{ and } M_{fixp}^{b} > 0 \\
    E_{int}^{b} + 1, & \text{if } S_{fp32}^{b} \text{ is a subnormal number and } M_{fixp}^{b} > 0.5 \\
    E_{int}^{b}, & \text{otherwise}
  \end{cases}
  $$

  Compute the block scale factor:

  $$
  S_{ue8m0}^{b}=2^{E_{int}^{b}}
  $$

  Compute the block conversion factor:

  $$
  R_{fp32}^{b}=\frac{1}{fp32(S_{ue8m0}^{b})}
  $$

  Perform quantization on each element in the block:

  $$
  p_i=cast\_to\_dst\_type(d^{i}\times R_{fp32}^{b},round_mode)
  $$

  Output the scale and quantization result corresponding to the block:

  $$
  (S_{ue8m0}^{b},[p_i]_{i=1}^{k})
  $$

## Parameters

<table><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
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
      <td>Input tensor to be quantized, corresponding to the input data D in the formula.</td>
      <td><ul><li>The shape is [M, N].</li><li>N must be greater than 0 and divisible by 64.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>Input</td>
      <td>Quantization group index, which describes the boundaries of groups in cumsum form.</td>
      <td><ul><li>Each element represents the end row index of a group.</li><li>The index values must be non-negative and non-decreasing.</li><li>The last element must be equal to M.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>Result of quantizing x along the -1 axis.</td>
      <td><ul><li>The shape must be the same as x, that is, [M, N].</li><li>The data type must be the same as dst_dtype.</li></ul></td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y1_scale</td>
      <td>Output</td>
      <td>Quantization scale corresponding to each group along the -1 axis.</td>
      <td><ul><li>The shape is [M, ceil(N/64), 2].</li><li>Every 2 scales in the last dimension are stored in pairs; one scale is generated for every 32 elements along the N direction, and even padding is performed, where the pad value is 0.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Result of quantizing x along the -2 axis.</td>
      <td><ul><li>The shape must be the same as x, that is, [M, N].</li><li>The data type must be the same as dst_dtype.</li></ul></td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2_scale</td>
      <td>Output</td>
      <td>Quantization scale corresponding to each group along the -2 axis.</td>
      <td><ul><li>The shape is [floor(M/64)+G, N, 2], where G is the number of elements in group_index.</li><li>One scale is generated for every 32 elements along the second-to-last dimension, and even padding is performed.</li><li>Every two rows of data in the y2_scale output need to be interleaved.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Optional attribute</td>
      <td>Rounding mode during quantization conversion, corresponding to round_mode in the formula.</td>
      <td><ul><li>Only "rint" is supported, with the default value "rint".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>Optional attribute</td>
      <td>Computation method of y1_scale and y2_scale.</td>
      <td><ul><li>Only the value 1 is supported, with the default value 1.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_dtype</td>
      <td>Optional attribute</td>
      <td>Target output data types of y1 and y2.</td>
      <td><ul><li>The value 35 indicates FLOAT8_E5M2, the value 36 indicates FLOAT8_E4M3FN, and the default value is 36.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_dtype_value</td>
      <td>Optional attribute</td>
      <td>Reserved parameter.</td>
      <td><ul><li>Only 0.0 is supported, with the default value 0.0.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The last dimension N of the input x must be divisible by 64.
- The shapes of y1 and y2 must be the same as x, both being [M, N]; the data types of y1 and y2 must be the same as the target type specified by dst_dtype.
- The shape of y1_scale is [M, ceil(N/64), 2]; the shape of y2_scale is [floor(M/64)+G, N, 2], where G is the number of elements in group_index.
- The last element of group_index must be equal to x.shape[0]. The cumsum mode is used, and each value represents the accumulated number of rows of the corresponding group.

## Invocation Description

| Invocation method | Invocation sample | Description |
| :------- | :------- | :--- |
| aclnn API | [test_aclnn_grouped_dynamic_mx_quant_with_dual_axis](./examples/arch35/test_aclnn_grouped_dynamic_mx_quant_with_dual_axis.cpp) | Calls the GroupedDynamicMxQuantWithDualAxis operator through the [aclnnGroupedDynamicMxQuantWithDualAxis](./docs/aclnnGroupedDynamicMxQuantWithDualAxis_en.md) API. |
| PyTorch API | [torchapi_grouped_dynamic_mx_quant_with_dual_axis](./docs/torchapi_grouped_dynamic_mx_quant_with_dual_axis_en.md) | Calls the GroupedDynamicMxQuantWithDualAxis operator through the [grouped_dynamic_mx_quant_with_dual_axis](./docs/torchapi_grouped_dynamic_mx_quant_with_dual_axis_en.md) API. |
