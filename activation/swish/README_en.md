# Swish

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: Computes the Swish activation function value.

- Formula:
  $$
  y = \frac{x}{1 + e^{-\text{scale} \times x}}
  $$
  Here:
  - $x$ denotes the input tensor
  - $\text{scale}$ denotes the scaling parameter, with the default value 1.0
  - $e$ denotes the base of the natural logarithm
  - $y$ denotes the output tensor

## Parameters

<table style="undefined;table-layout: fixed; width: 1480px">
  <colgroup>
    <col style="width: 177px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 292px">
    <col style="width: 152px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Represents the input tensor, supporting 1D-8D dimensions.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor, which has the same type, shape, and format as the input x.</td>
      <td>Same as x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Attribute</td>
      <td>Represents the scaling parameter, which adjusts the shape of the Swish function. Default value: 1.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_swish](./examples/test_aclnn_swish.cpp) | Calls the Swish operator through the [aclnnSwish](./docs/aclnnSwish_en.md) API.    |
