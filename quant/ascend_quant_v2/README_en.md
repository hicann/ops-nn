# AscendQuantV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>kirin X90 processor series products</term> | √ |
|  <term>kirin 9030 processor series products</term> | √ |

## Function

- Operator function: performs a quantization operation on the input x. You can set axis to specify the axes that scale and offset correspond to. The shapes of scale and offset must be equal to the axes of x specified by axis or be 1. Currently, axis supports setting the last two dimensions.
- Formula:
  - When sqrt\_mode is false, the formula is:

    $$
    y = round((x * scale) + offset)
    $$

  - When sqrt\_mode is true, the formula is:

    $$
    y = round((x * scale * scale) + offset)
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
      <td>The input on which quantization is performed, corresponding to `x` in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>The scale value in quantization, corresponding to `scale` in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Optional input</td>
      <td>The offset value in dequantization, corresponding to `offset` in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sqrt_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the logic for scale participating in the computation, corresponding to `sqrtMode` in the formula.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the conversion method for casting to the output data type. Supported values: round, ceil, trunc, and floor.</li><li>The default value is "round".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the data type of the output.</li><li>The default value is DT_INT8.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the dimension of x that scale and offset correspond to.</li><li>The default value is -1.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the result output `y` of AscendQuantV2, corresponding to `y` in the formula. The shape is consistent with the shape of the input `x`.</td>
      <td>INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas inference products</term>:
  - Data type:
    - The input tensor data type does not support BFLOAT16.
    - The output tensor data type supports only INT8.
  - round_mode: supported values are round, ceil, trunc, and floor.
  - dst_type: supports the value 2, indicating INT8.
  - axis: supports only the last dimension of x (assuming the number of dimensions of the input x is xDimNum, the value of axis is -1 or xDimNum-1).
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
  - The output tensor data type supports only INT8 and INT4.
  - dst_type: supports the values 2 and 29, indicating INT8 and INT4, respectively.
  - axis: supports the last two dimensions of x (assuming the number of dimensions of the input x is xDimNum, the value range of axis is [-2, -1] or [xDimNum-2, xDimNum-1]).
- <term>Ascend 950PR&950DT products</term>:
   - round_mode: when dst_type is FLOAT8_E5M2 or FLOAT8_E4M3FN, only round is supported. When dst_type is HIFLOAT8, round and hybrid are supported. When dst_type is another type, round, ceil, trunc, and floor are supported.
   - axis: supports the last two dimensions of x (assuming the number of dimensions of the input x is xDimNum, the value range of axis is [-2, -1] or [xDimNum-2, xDimNum-1]).

- Kirin X90/kirin 9030 processor series products: `x`, `scale`, and `offset` do not support BFLOAT16. The data type of `y` does not support INT4, HIFLOAT8, FLOAT8_E5M2, or FLOAT8_E4M3FN.

## Constraints

None

## Invocation Description

| Invocation method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_ascend_quant](examples/arch22/test_aclnn_ascend_quant.cpp) | Invokes the AscendQuantV2 operator through the [aclnnAscendQuant](docs/aclnnAscendQuant_en.md) API. |
| aclnn API  | [test_aclnn_ascend_quant_v3](examples/test_aclnn_ascend_quant_v3.cpp) | Invokes the AscendQuantV2 operator through the [aclnnAscendQuantV3](docs/aclnnAscendQuantV3_en.md) API. |
| Graph mode | -  | Invokes the AscendQuantV2 operator through graph construction using the [operator IR](op_graph/ascend_quant_v2_proto.h).         |
