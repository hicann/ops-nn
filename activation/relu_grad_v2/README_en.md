# ReluGradV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: the backpropagation gradient corresponding to the Relu operation.

- Formula:
      $$
      gradients = backprops *
      \begin{cases}
      1, \quad mask > 0\\
      0,  \quad mask \leq 0
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
      <td>Backpropagation gradient passed to the corresponding Relu operation</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Input</td>
      <td>Feature passed as input to the corresponding ReluV2 operation</td>
      <td>UINT1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops</td>
      <td>Output</td>
      <td>Output tensor in the formula</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                            |
| ---------------- | --------------------------- |-----------------------------------------------|
| Graph mode | [test_geir_relu_grad_v2.cpp](./examples/test_geir_relu_grad_v2.cpp)  | Calls the ReluGradV2 operator by constructing a graph through the [operator IR](./op_graph/relu_grad_v2_proto.h). |
