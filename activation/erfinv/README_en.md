# Erfinv

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: erfinv is the inverse function of the Gaussian error function erf. The operator returns the quantile of each element in the input tensor under the standard normal distribution function.

- Formula:

$$y=\mathrm{erfinv}(x)$$

Here $x=erf(y)=\frac{2}{\sqrt{\pi}} \int_{0}^{y} e^{-t^2} dt$

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
      <td>Input parameter for the erfinv computation, corresponding to x in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter of the erfinv computation, corresponding to y in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas training products</term>: The supported data types are FLOAT and FLOAT16.

## Constraints

- Input shape restriction: the input dimensions cannot exceed 8D, and only 0D to 8D are supported.

## Invocation Description

| Invocation method | Invocation sample | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_erfinv](./examples/test_aclnn_erfinv.cpp) | Invoke the Erfinv operator through the [aclnnErfinv](./docs/aclnnErfinv&aclnnInplaceErfinv_en.md) API. |
