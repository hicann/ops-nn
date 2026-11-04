# ForeachLerpList

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Description: Performs interpolation computation on the elements at corresponding positions of two tensor lists, where the tensor list weight is the interpolation coefficient.
- Formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  weight = [{weight_0}, {weight_1}, ... {weight_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = x1_i + {\rm weight}_i × (x2_i - x1_i) (i=0,1,...n-1)
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
      <td>x1</td>
      <td>Input</td>
      <td>Denotes the first input tensor list for linear interpolation computation, corresponding to <code>x1</code> in the formula. Empty tensors are supported. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Denotes the second input tensor list for linear interpolation computation, corresponding to <code>x2</code> in the formula. Empty tensors are supported. The data type, data format, and shape are consistent with those of the input parameter <code>x1</code>. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Denotes the interpolation coefficient tensor list for linear interpolation computation, corresponding to <code>weight</code> in the formula. Empty tensors are supported. The data type, data format, and shape are consistent with those of the input parameter <code>x1</code>. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the output tensor list for linear interpolation computation, corresponding to <code>y</code> in the formula. Empty tensors are supported. The data type and data format are consistent with those of the input parameter <code>x1</code>, the shape is consistent with that of the input parameter <code>x1</code>, and the storage shape size is not less than the storage shape size of the input parameter <code>x1</code>. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- Non-contiguous tensors: only the aclnn interface supports non-contiguous input and output. The interface first converts non-contiguous tensors to contiguous tensors for computation, and then writes the output result back to the original output view through ViewCopy. The graph mode (IR) and kernel direct-invocation paths do not support non-contiguous tensors.
- The dimensions of x1, x2, weight, and y range from 0 to 8.
- The shape of out must be consistent with that of x1, and the storage shape size of out must not be less than that of x1.
- The maximum length of a TensorList is 256.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_foreach_lerp_list](examples/test_aclnn_foreach_lerp_list.cpp) | Invokes the ForeachLerpList operator through the [aclnnForeachLerpList](docs/aclnnForeachLerpList_en.md) interface. |
| Graph mode | -  | Invokes the ForeachLerpList operator by building a graph through the [operator IR](op_graph/foreach_lerp_list_proto.h).         |
