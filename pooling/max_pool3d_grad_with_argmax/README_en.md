# MaxPool3DGradWithArgmax

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

Operator function: Backpropagation of forward max pooling. The operator fills the gradients back to the coordinates of the maximum value of each window, and accumulates at the same coordinates.

## Parameters

<table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 300px">
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
      <td>Input parameter for the MaxPool3DGradWithArgmax computation, indicating the input tensor of the forward pass.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Input parameter for the MaxPool3DGradWithArgmax computation, indicating the gradient of the current node.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>Input</td>
      <td>Indicates the index position of the maximum element in the forward input. Its data format must be consistent with `x`, and its shape must be consistent with `grad`.</td>
      <td>INT32</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Indicates the window size of max pooling.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Indicates the strides of the pooling operation.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Attribute</td>
      <td>Indicates the number of layers padded with 0 by pads in the D, H, and W directions of the input.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>Optional attribute</td>
      <td>Indicates the stride of the elements in the window.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>Optional attribute</td>
      <td>Indicates whether the deduced output shape in the forward max pooling process is rounded up.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter of the MaxPool3DGradWithArgmax computation. Its shape and data format must be consistent with `x`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
  </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_max_pool3d_with_argmax_backward.cpp](examples/test_aclnn_max_pool3d_with_argmax_backward.cpp) | Calls the MaxPool3DGradWithArgmax operator through the [aclnnMaxPool3dWithArgmaxBackward](docs/aclnnMaxPool3dWithArgmaxBackward_en.md) interface. |
