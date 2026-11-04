# AdaptiveAvgPool2dGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|×|
|Atlas A2 products|×|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|×|
|Atlas training products|×|

## Function

- Operator function: the backward computation of AdaptiveAvgPool2d.

## Parameters

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 250px">
  <col style="width: 200px">
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
      <td>input_grad</td>
      <td>Input</td>
      <td>The input parameter for the AdaptiveAvgPool2dGrad computation, representing the gradient of the current node. The data type, data format, and total number of shape dimensions are consistent with the input parameter `orig_input_shape`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>orig_input_shape</td>
      <td>Input</td>
      <td>The input parameter for the AdaptiveAvgPool2dGrad computation. The data type, data format, and total number of shape dimensions are consistent with the input parameter `input_grad`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_grad</td>
      <td>Output</td>
      <td>The output parameter of the AdaptiveAvgPool2dGrad computation. The data type and shape are consistent with the input parameter `orig_input_shape`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_adaptive_avg_pool2d_backward.cpp](../adaptive_avg_pool2d_grad/examples/test_aclnn_adaptive_avg_pool2d_backward.cpp) | Invoke the AdaptiveAvgPool2dGrad operator through the [aclnnAdaptiveAvgPool2dBackward](../adaptive_avg_pool2d_grad/docs/aclnnAdaptiveAvgPool2dBackward_en.md) interface. |
