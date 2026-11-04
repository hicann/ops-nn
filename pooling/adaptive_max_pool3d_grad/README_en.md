# AdaptiveMaxPool3DGrad

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

- Operator function: backpropagation of the forward adaptive max pooling, which backfills the gradients to the coordinates of the maximum values of each adaptive window, and accumulates them at identical coordinates.

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
      <td>x</td>
      <td>Input</td>
      <td>Input parameter for the AdaptiveMaxPool3DGrad computation.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Input parameter for the AdaptiveMaxPool3DGrad computation, denoting the gradient of the current node.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>Input</td>
      <td>Input parameter for the AdaptiveMaxPool3DGrad computation. It denotes the index positions of the maximum elements in the forward input. The data format and shape are consistent with those of input parameter `grad`.</td>
      <td>INT32</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter of the AdaptiveMaxPool3DGrad computation. The data format and shape are consistent with those of input parameter `x`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_adaptive_max_pool3d_backward.cpp](examples/test_aclnn_adaptive_max_pool3d_backward.cpp) | Invokes the AdaptiveMaxPool3DGrad operator through the [aclnnAdaptiveMaxPool3dBackward](docs/aclnnAdaptiveMaxPool3dBackward_en.md) interface. |
