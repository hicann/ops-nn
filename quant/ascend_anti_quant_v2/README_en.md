# AscendAntiQuantV2

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: Dequantizes the input x according to the input scale and offset.

- Formula:
  - When sqrt\_mode is true and offset is None, the formula is:

    $$
    y = cast\_to\_dst\_type((x) * scale * scale)
    $$

  - When sqrt\_mode is true and offset is not None, the formula is:

    $$
    y = cast\_to\_dst\_type((x + offset) * scale * scale)
    $$

  - When sqrt\_mode is false and offset is None, the formula is:

    $$
    y = cast\_to\_dst\_type((x) * scale)
    $$

  - When sqrt\_mode is false and offset is not None, the formula is:

    $$
    y = cast\_to\_dst\_type((x + offset) * scale)
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
      <td><ul><li>Represents the input tensor of the operator, corresponding to x in the formulas;</li><li>Empty tensors are not supported;</li><li>When the data type is INT4, the tail axis of the shape is an even number.</li></ul></td>
      <td>INT4, INT8, HIFLOAT8, FLOAT8_E4M3, FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td><ul><li>Represents the scale value in dequantization, corresponding to scale in the formulas;</li><li>Empty tensors are not supported;</li><li>The number of dimensions of scale must be the same as that of x, or 1;</li><li>If x is 1-dimensional, the shape of scale must be [1] or the same as x;</li><li>If scale is 1-dimensional, its size must be 1, x[-1], or x[-2];</li><li>If scale is multi-dimensional, it can have at most one dimension that is not 1, and this non-1 dimension can only be the -1 or -2 axis;</li><li>When the type of the input x is HIFLOAT8, FLOAT8_E4M3, or FLOAT8_E5M2, scale supports only FLOAT32.</li></ul></td>
      <td>FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Optional Input</td>
      <td><ul><li>Represents the offset value in dequantization, corresponding to offset in the formulas;</li><li>Empty tensors are not supported;</li><li>The data type and shape must be consistent with those of scale.</li></ul></td>
      <td>Same as scale</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Optional Attribute</td>
      <td><ul><li>Represents the data type of the output;</li><li>Supported values are 1 and 27, representing FLOAT16 and BFLOAT16 respectively.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sqrt_mode</td>
      <td>Optional Attribute</td>
      <td><ul><li>Represents the logic of scale participation in the computation, corresponding to sqrt_mode in the formulas;</li><li>When the data type of x is HIFLOAT8, FLOAT8_E4M3, or FLOAT8_E5M2, sqrt_mode is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Represents the computation output of dequantization, corresponding to y in the formulas;</li><li>The shape is the same as that of the input x.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- <term>Atlas inference products</term>:
  - Data types:
    - The input `x` supports only INT8.
    - The inputs `scale` and `offset` and the output `y` do not support BFLOAT16.
  - The input `dstType` supports only the value 1, representing FLOAT16.
  - The inputs `scale` and `offset` support only one dimension, which can only equal the tail axis size of x or 1; when x is of the int32 type, it must be 8 times the tail axis size of x.
- <term>Atlas A2 products/Atlas A3 products</term>:
  - The input `x` supports only INT4 and INT8.
  - The inputs `scale` and `offset` support only one dimension, which can only equal the tail axis size of x or 1; when x is of the int32 type, it must be 8 times the tail axis size of x.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_ascend_anti_quant](examples/test_aclnn_ascend_anti_quant.cpp) | Invokes the AscendAntiQuantV2 operator through the [aclnnAscendAntiQuant](docs/aclnnAscendAntiQuant_en.md) interface. |
| Graph-mode invocation | -  | Builds the graph and invokes the AscendAntiQuantV2 operator through the [operator IR](op_graph/ascend_anti_quant_v2_proto.h).         |
