# ReluV2

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

- Operator function: Activation function that returns a tensor with the same shape as the input tensor. When a value in the tensor is greater than or equal to 0, the value is taken; when it is less than 0, 0 is taken.

- Formula:

$$
relu_v2(self) = \begin{cases} self, & self\gt 0 \\ 0, & self\le 0 \end{cases}
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
      <td>Input parameter of the ReluV2 computation, which is the input self in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT32, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output of the ReluV2 computation, which is relu_v2(self) in the formula.</td>
      <td>FLOAT, FLOAT16, INT8, INT32, INT64, BFLOAT16, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Output</td>
      <td>Output of the ReluV2 computation. When the input value is greater than 0, the value is 1; when the input value is less than or equal to 0, the value is 0.</td>
      <td>UINT1</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_relu_v2](./examples/test_geir_relu_v2.cpp)   | Invokes the ReluV2 operator through graph construction with the [operator IR](./op_graph/relu_v2_proto.h). |
