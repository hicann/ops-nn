# DynamicQuantV2

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: performs pertoken/pertensor/perchannel symmetric/asymmetric dynamic quantization for the input tensor. In the MOE scenario, the smooth_scales of each expert are different, which are distinguished by the input group_index.

- Formula:
  - Symmetric quantization:
    - If smooth_scales is not provided:

      $$
        scale=row\_max(abs(x))/dtypeMax
      $$

      $$
        y=round(x/scale)
      $$

    - If smooth_scales is provided:

      $$
        input = x\cdot smooth_scales
      $$

      $$
        scale=row\_max(abs(input))/dtypeMax
      $$

      $$
        y=round(input/scale)
      $$

  - Asymmetric quantization:
    - If smooth_scales is not provided:

      $$
        scale=(row\_max(x) - row\_min(x))/(dtypeMax - dtypeMin)
      $$

      $$
        offset=dtypeMax-row\_max(x)/scale
      $$

      $$
        y=round(x/scale+offset)
      $$

    - If smooth_scales is provided:

      $$
        input = x\cdot smooth_scales
      $$

      $$
        scale=(row\_max(input) - row\_min(input))/(dtypeMax - dtypeMin)
      $$

      $$
        offset=dtypeMax-row\_max(input)/scale
      $$

      $$
        y=round(input/scale+offset)
      $$

  Where:

  - row_max represents the computation of the maximum value of each row.
  - row_min represents the computation of the minimum value of each row.
  - dtypeMax is the maximum value of the output data type.
  - dtypeMin is the minimum value of the output data type.

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td><ul><li>The input tensor of the operator, corresponding to `x` in the formula.</li><li>The number of dimensions of the shape must be greater than 1.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scales</td>
      <td>Optional input</td>
      <td><ul><li>The input tensor of the operator, corresponding to `smooth_scales` in the formula.</li><li>The data type is the same as the data type of x. When group_index is empty, the shape is 1D and Dim[0] is the last dimension of x. When group_index is not empty, the shape is 2D. Dim[0] is the number of experts (E). E must not be greater than 1024, and it must not be less than 1 when x is not empty. Dim[1] is the last dimension of x.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>group_index</td>
      <td>Optional input</td>
      <td><ul><li>Specifies the index of groups.</li><li>1D shape is supported, which is [E, ]. The first dimension of the shape is the same as the first dimension of the shape of smooth_scales.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the data type of the output y. DT_INT4(29), DT_INT8(2), DT_FLOAT8_E5M2(35), DT_FLOAT8_E4M3FN(36), and DT_HIFLOAT8(34) are supported.</li><li>The default value is DT_INT8(2).</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_symmetrical</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies whether to use symmetric quantization.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the quantization mode. "pertoken", "pertensor", and "perchannel" are supported.</li><li>The default value is "pertoken".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the range of the quantization output. It takes effect only in the Hifloat8 scenario. The supported value range is 0~32768.</li><li>The default value is 0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>The output tensor after quantization, corresponding to `y` in the formula.</li><li>The shape is the same as the input x. When the data type of y is INT4, the last dimension of x must be divisible by 2.</li></ul></td>
      <td>INT8, INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Output</td>
      <td><ul><li>The scaling coefficient after quantization, corresponding to `scale` in the formula.</li><li>When quant_mode is "pertoken", the shape is the same as the shape of x with the last dimension removed. When quant_mode is "pertensor", the shape is (1,).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Output</td>
      <td><ul><li>The offset used for asymmetric quantization, corresponding to `offset` in the formula.</li><li>The shape is the same as the shape of scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas inference products</term> and <term>Atlas training products</term>:
  - Input `x`: only the FLOAT16 data type is supported.
  - Output `y`: only the INT8 data type is supported.
  - The inputs `smooth_scales` and `group_index` are reserved parameters and do not participate in the computation in the current version.
  - Input `dst_type`: only value 2 can be configured.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: only the INT8 and INT4 data types are supported for the output `y`.

- Kirin X90/Kirin 9030 processor series products:
  - Input `x`: the BFLOAT16 data type is not supported.
  - Optional input `smooth_scales`: the BFLOAT16 data type is not supported.
  - Output `y`: the data types INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, and HIFLOAT8 are not supported.

## Constraints

- E should not be greater than the product of the dimensions of x after the last dimension is removed (S).
- The values of `group_index` should increase, in the range from 0 to S. The last value should be S. Otherwise, the result is meaningless.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_dynamic_quant_v2](examples/test_aclnn_dynamic_quant_v2.cpp) | Invoke the DynamicQuantV2 operator through the [aclnnDynamicQuantV2](docs/aclnnDynamicQuantV2_en.md) interface. |
| Graph mode | -  | Invoke the DynamicQuantV2 operator by constructing a graph through the [operator IR](op_graph/dynamic_quant_v2_proto.h).         |
