# GroupQuant

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>kirin X90 processor series products</term> | √ |
|  <term>kirin 9030 processor series products</term> | √ |

## Function

- Operator function: Performs the grouped quantization operation on the input x.
- Formula:

  $$
  y = round((x * scale) + offsetOptional)
  $$

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
      <td>Input on which quantization needs to be performed, which corresponds to `x` in the formula. The shape supports 2 dimensions.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Scale value in the quantization, which corresponds to `scale` in the formula. The shape supports 2 dimensions, and the second dimension equals the second dimension of the x shape.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>Input</td>
      <td>GroupIndex value in the quantization. The shape supports 1 dimension, and the dimension equals the first dimension of the scale shape.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Optional input</td>
      <td>Offset value in the quantization, which corresponds to `offsetOptional` in the formula. The data type is the same as that of `scale`. `offsetOptional` is 1 number.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the data type of the output. Supported values are 2 and 29, which indicate INT8 and INT4 respectively.</li><li>Default value: DT_INT8.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output after grouped quantization, which corresponds to y in the formula. The shape is the same as the shape of `x`.</td>
      <td>INT4, INT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraints

- The data type of the input `scale` is the same as that of the input `offset`.
- If the attribute `dst_type` is 29 (INT4), the last dimension of the shape of the output `y` must be divisible by 2.
- The input `group_index` must be a non-decreasing sequence, and the maximum value must equal the size of the first dimension of the shape of the input `x`.
- The shape of the input `offset` currently supports only [1, ] or [].

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_group_quant](examples/test_aclnn_group_quant.cpp) | Calls the GroupQuant operator through the [aclnnGroupQuant](docs/aclnnGroupQuant_en.md) API. |
| Graph mode | -  | Calls the GroupQuant operator through graph construction with the [operator IR](op_graph/group_quant_proto.h).         |
