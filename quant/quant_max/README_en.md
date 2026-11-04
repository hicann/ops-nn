# QuantMax

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: quantizes input x according to the input scale, and computes the maximum value amax of the absolute value of input x.

- Formula:

  $$
  y = \text{Cast}(x \times scale, \text{dst\_type}, \text{round\_mode})
  $$

  $$
  amax = \max(|x|)
  $$

  where $x$ denotes the input tensor, $scale$ denotes the quantization scaling factor, $\text{dst\_type}$ specifies the output type, $\text{round\_mode}$ specifies the rounding mode, and $amax$ is the maximum value of the absolute values of all elements of the input tensor.

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>Input tensor to be quantized, corresponding to x in the formula. Any shape of 1-8 dimensions is supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Quantization scaling factor, corresponding to scale in the formula. The shape is fixed to [1].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Quantized output tensor, corresponding to y in the formula. Its shape is the same as x, and its data type is determined by dst_type.</td>
      <td>HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>amax</td>
      <td>Output</td>
      <td>Maximum value of the absolute value of x, corresponding to amax in the formula. The shape is fixed to [1], and the data type is the same as x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Attribute</td>
      <td>The value must be a valid rounding mode supported by the Cast API.</td>
      <td>string</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Attribute</td>
      <td>Specifies the output quantization data type. 34=HIFLOAT8, 35=FLOAT8_E5M2, 36=FLOAT8_E4M3.</td>
      <td>int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- Matching relationship between `round_mode` and `dst_dtype`:
    - When `dst_dtype` is `35` or `36`, `round_mode` must be `"rint"`.
    - When `dst_dtype` is `34`, `round_mode` must be `"round"` or `"hybrid"`.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_quant_max](./examples/arch35/test_aclnn_quant_max.cpp) | Invokes the QuantMax operator through the [aclnnQuantMax](./docs/aclnnQuantMax_en.md) interface.    |
| Graph-mode invocation | -   | Invokes the QuantMax operator through the [operator IR](./op_graph/quant_max_proto.h) graph construction method. |
