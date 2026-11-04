# LeakyReluGrad

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

Description: Backward of the [aclnnLeakyRelu](../leaky_relu/docs/aclnnLeakyRelu&aclnnInplaceLeakyRelu_en.md) activation function.
Formula:

$$
output =
\begin{cases}
gradOutput, &if\ self \gt 0 \\
gradOutput*negativeSlope, &if\ self \le 0
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
      <td>gradOutput</td>
      <td>Input</td>
      <td>Gradient.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>self</td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>negativeSlope</td>
      <td>Input</td>
      <td>Slope when self < 0.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT8, BOOL, INT16, UINT8, BFLOAT16</td>
      <td>-</td>
    </tr>
     <tr>
      <td>selfIsResult</td>
      <td>Input</td>
      <td>When selfIsResult is true, negativeSlope cannot be a negative number.</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>out</td>
      <td>Output</td>
      <td>Computation output.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, DOUBLE</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas inference products</term> and <term>Atlas training products</term>: The data types of gradOutput, self, and out support FLOAT, FLOAT16, and DOUBLE. The data type of negativeSlope supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT8, BOOL, INT16, and UINT8.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code                                                                         | Description                                                                               |
| ---------------- |------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| aclnn API  | [test_aclnn_leaky_relu_backward.cpp](examples/test_aclnn_leaky_relu_backward.cpp) | Calls the LeakyReluGrad operator through the [aclnnLeakyReluBackward](docs/aclnnLeakyReluBackward_en.md) API. |
