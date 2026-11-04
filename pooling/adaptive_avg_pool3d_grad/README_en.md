# AdaptiveAvgPool3dGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|×|
|Atlas training products|×|

## Function

- Operator function: Backward computation of AdaptiveAvgPool3d.

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
      <td>y_grad</td>
      <td>Input</td>
      <td>Input parameter for the AdaptiveAvgPool3dGrad computation, representing the gradient of the current node. The data type, data format, and total number of shape dimensions are consistent with the input parameter `x`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input parameter for the AdaptiveAvgPool3dGrad computation. The data type, data format, and total number of shape dimensions are consistent with the input parameter `y_grad`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x_grad</td>
      <td>Output</td>
      <td>Output parameter of the AdaptiveAvgPool3dGrad computation. The data type and shape are consistent with the input parameter `x`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_adaptive_avg_pool3d_backward.cpp](examples/test_aclnn_adaptive_avg_pool3d_backward.cpp) | Invoke the AdaptiveAvgPool3d operator through the [aclnnAdaptiveAvgPool3dBackward](docs/aclnnAdaptiveAvgPool3dBackward_en.md) API. |
