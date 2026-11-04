# MaxPoolWithArgmaxV3

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>                   |     √    |
|  <term>Atlas A3 products</term>  |     ×    |
|  <term>Atlas A2 products</term>   |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>                   |     ×    |
|  <term>Atlas inference products</term>                           |     ×    |
|  <term>Atlas training products</term>                           |     ×    |

## Function

- Operator function: computes the 2D max pooling operation on the input data.

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
      <td>ksize</td>
      <td>Input attribute</td>
      <td>Size of the pooling window</td>
      <td>Int</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Input attribute</td>
      <td>Stride of the window movement</td>
      <td>Int</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Input attribute</td>
      <td>Number of layers padded on each edge</td>
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
      <td>Controls the element stride within the window</td>
      <td>Int</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>Input attribute</td>
      <td>When it is true, the ceil rounding method is used</td>
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
    <tr>
      <td>argmax</td>
      <td>Output</td>
      <td>Position index of the maximum value within each pooling window; the shape is the same as y</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Sample code                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_max_pool_with_argmax_v3](examples/arch35/test_aclnn_max_pool_with_argmax_v3.cpp) | Calls the MaxPoolWithArgmaxV3 operator through the [aclnnMaxPool2dWithIndices](../max_pool3d_with_argmax_v2/docs/aclnnMaxPool2dWithIndices_en.md) interface. |
