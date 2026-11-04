# MaxPoolGradWithArgmaxV3

English | [简体中文](./README.md)


## Applicable Products

| Product | Support |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>                   |     √    |
|  <term>Atlas A3 products</term>  |     ×    |
|  <term>Atlas A2 products</term>   |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>                   |     ×    |
|  <term>Atlas inference products</term>                           |     ×    |
|  <term>Atlas training products</term>                           |     ×    |

## Function

- Operator function: the backward of the forward max pooling MaxPoolGradWithArgmaxV3.

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
      <td>Input x</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Gradient tensor</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>Input</td>
      <td>Input grad</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Input attribute</td>
      <td>Pooling window size</td>
      <td>Int</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Input attribute</td>
      <td>Window movement stride</td>
      <td>Int</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Input attribute</td>
      <td>Number of padding layers on each edge</td>
      <td>Int</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Input attribute</td>
      <td>Data type of the output argmax</td>
      <td>Int</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>Input attribute</td>
      <td>Controls the element stride in the window</td>
      <td>Int</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>Input attribute</td>
      <td>When the value is true, the rounding-up (ceil) method is used</td>
      <td>Bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Input attribute</td>
      <td>Data format</td>
      <td>String</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_max_pool_grad_with_argmax_v3](examples/arch35/test_aclnn_max_pool_grad_with_argmax_v3.cpp) | Calls the aclnnMaxPoolGradWithArgmaxV3 operator through the [aclnnMaxPool2dWithIndicesBackward](../max_pool3d_grad_with_argmax/docs/aclnnMaxPool2dWithIndicesBackward_en.md) interface.|
