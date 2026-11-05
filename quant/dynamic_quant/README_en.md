# DynamicQuant

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |
| <term>KirinX90 processor series products</term> | √ |
| <term>Kirin9030 processor series products</term> | √ |

## Function

- Operator function: performs per-token symmetric dynamic quantization for the input tensor.

- Formula:
  - If smoothScalesOptional is not provided:

  $$
   scaleOut=row\_max(abs(x))/dtypeMax
  $$

  $$
   yOut=round(x/scaleOut)
  $$

  - If smoothScalesOptional is provided:

  $$
  input = x\cdot smoothScalesOptional
  $$

  $$
   scaleOut=row\_max(abs(input))/dtypeMax
  $$

  $$
   yOut=round(input/scaleOut)
  $$

  Here, row\_max denotes taking the maximum value of each row, and dtypeMax is the maximum value of the output data type.

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
      <td><ul><li>Tensor input of the operator, corresponding to `x` in the formula.</li><li>The shape must have more than 1 dimension.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scales</td>
      <td>Optional input</td>
      <td><ul><li>Tensor input of the operator, corresponding to `smoothScalesOptional` in the formula.</li><li>The shape length is the same as the last dimension of `x`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>group_index</td>
      <td>Optional input</td>
      <td><ul><li>Index of the specified group.</li><li>A 1D shape is supported, which is [E, ], and the shape is the same as the first dimension of the shape of output `scale`.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the data type of output y. DT_INT4(29), DT_INT8(2), DT_FLOAT8_E5M2(35), DT_FLOAT8_E4M3FN(36), and DT_HIFLOAT8(34) are supported.</li><li>The default value is DT_INT8(2).</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Quantized output tensor, corresponding to `yOut` in the formula.</li><li>The data type is consistent with the data type configured by parameter `dst_type`, and the shape is consistent with the shape of input `x`.</li></ul></td>
      <td>INT8, INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Output</td>
      <td>Scale factor after quantization, corresponding to `scaleOut` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas inference products</term>, <term>Atlas training products</term>:
  - Inputs `x` and `smooth_scales`: only the FLOAT16 data type is supported.
  - Output `y`: only the INT8 data type is supported.
  - Input `smooth_scales` is a reserved parameter and does not participate in the computation in the current version.

- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: the data types of output `y` support only INT8 and INT4.

- KirinX90/Kirin9030 processor series products:
  - Inputs `x` and `smooth_scales`: BFLOAT16 is not supported.
  - Output `y`: the data types INT4, FLOAT8_E5M2, FLOAT8_E4M3FN, and HIFLOAT8 are not supported.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_dynamic_quant](examples/test_aclnn_dynamic_quant.cpp) | Invokes the DynamicQuant operator through the [aclnnDynamicQuant](docs/aclnnDynamicQuant_en.md) interface. |
| Graph-mode invocation | -  | Invokes the DynamicQuant operator through graph construction with the [operator IR](op_graph/dynamic_quant_proto.h).         |
