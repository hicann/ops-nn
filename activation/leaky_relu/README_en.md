# LeakyRelu

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | √ |

## Function

- Description: an activation function that solves the problem where the Relu function outputs 0 when the input is less than 0, avoiding the situation where neurons cannot update parameters.
- Formula:

  $$
  out = max(0,self) + negativeSlope * min(0,self)
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 813px"><colgroup>
  <col style="width: 103px">
  <col style="width: 144px">
  <col style="width: 194px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>self</td>
      <td>Input</td>
      <td>The input for the LeakyRelu activation function, the self in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>negativeSlope</td>
      <td>Input</td>
      <td>Denotes the slope when self < 0, the negativeSlope in the formula.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
      <tr>
      <td>out</td>
      <td>Output</td>
      <td>The output of the LeakyRelu activation function.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas training products</term>: the data types support FLOAT and FLOAT16.

## Constraints

- negativeSlope uses an integer type as the attribute input, while the input self is of the FLOAT type. If negativeSlope is greater than 2^24 or less than -2^24, precision loss may occur. Similarly, if the input self is of the FLOAT16 type, precision loss may occur when negativeSlope is greater than 2^11 or less than -2^11.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn API | [test_aclnn_leaky_relu.cpp](examples/test_aclnn_leaky_relu.cpp) | Calls the LeakyRelu operator through the [aclnnLeakyRelu&aclnnInplaceLeakyRelu](docs/aclnnLeakyRelu&aclnnInplaceLeakyRelu_en.md) API. |
| aclnn API | [test_aclnn_inplace_leaky_relu.cpp](examples/test_aclnn_inplace_leaky_relu.cpp) | Calls the LeakyRelu operator through the [aclnnLeakyRelu&aclnnInplaceLeakyRelu](docs/aclnnLeakyRelu&aclnnInplaceLeakyRelu_en.md) API. |
