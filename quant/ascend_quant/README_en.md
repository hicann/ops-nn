# AscendQuant

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: Performs quantization on the input x, and the sizes of scale and offset need to be the last dimension of x or 1.
- Formula:
  - When sqrtMode is false, the computation formula is:

    $$
    y = round((x * scale) + offset)
    $$

  - When sqrtMode is true, the computation formula is:

    $$
    y = round((x * scale * scale) + offset)
    $$

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
      <td>Input parameter for the ascend_quant computation, corresponding to x in the formula.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Input parameter for the ascend_quant computation, corresponding to scale in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td>Input parameter for the ascend_quant computation, corresponding to offset in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sqrt_mode</td>
      <td>Input</td>
      <td>Input parameter for the ascend_quant computation, corresponding to sqrt_mode in the formula.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Input</td>
      <td>Input parameter for the ascend_quant computation, corresponding to round_mode in the formula.</td>
      <td>STRING</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Input</td>
      <td>Input parameter for the ascend_quant computation, corresponding to dst_type in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter of the ascend_quant computation, corresponding to y in the formula.</td>
      <td>INT8, INT4, HIFLOAT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Ascend 950PR&950DT products </term>: The data types support FLOAT32 and FLOAT16.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | -  | Calls the AscendQuant operator by constructing a graph through the [operator IR](op_graph/ascend_quant_proto.h).         |
