# ReluGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: the backpropagation gradient corresponding to the Relu operation.

- Formula:

  $$
      gradInput = gradOutput *
      \begin{cases}
      1, \quad x > 0\\
      0,  \quad x \leq 0
      \end{cases}
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>gradients</td>
      <td>Input</td>
      <td>The backpropagation gradient passed to the corresponding Relu operation</td>
      <td>INT8, INT32, UINT8, INT64, FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>features</td>
      <td>Input</td>
      <td>The features passed as input to the corresponding Relu operation</td>
      <td>INT8, INT32, UINT8, INT64, FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops</td>
      <td>Output</td>
      <td>The output tensor in the formula</td>
      <td>INT8, INT32, UINT8, INT64, FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| Graph mode | [test_geir_relu_backward.cpp](examples/test_geir_relu_backward.cpp) | Calls the ReluGrad operator by building the graph through the [operator IR](op_api/relu_grad.h). |
