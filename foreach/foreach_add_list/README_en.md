# ForeachAddList

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

- Operator function: adds the elements of two tensor lists one by one, and the addition coefficient can be adjusted through the alpha parameter.

- Formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = x1_i + x2_i * alpha (i=0,1,...n-1)
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
      <td>Denotes the first input tensor list of the addition, corresponding to `x1` in the formula. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Denotes the second input tensor list of the addition, corresponding to `x2` in the formula. The data type, data format, and shape are consistent with those of the input parameter `x1`, and the data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Denotes the coefficient of the second input in the addition, corresponding to `alpha` in the formula. The number of elements is 1. The data type has a certain correspondence with the data type of the input parameter `x1`: when the data type of `x1` is FLOAT32, FLOAT16, or INT32, the data type is consistent with the data type of `x1`; when the data type of `x1` is BFLOAT16, the data type supports FLOAT32; when the data type of `x1` is INT16, INT8, or UINT8, the data type supports INT32.</td>
      <td>FLOAT32, FLOAT16, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the output tensor list of the addition, corresponding to `y` in the formula. The data type and data format are consistent with those of the input parameter `x1`, and the shape size is greater than or equal to the shape size of the input parameter `x1`. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: BFLOAT16, INT16, INT8, and UINT8 are not supported.

## Constraints

- The output does not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_foreach_add_list](examples/test_aclnn_foreach_add_list.cpp) | Invokes the ForeachAddList operator through the [aclnnForeachAddList](docs/aclnnForeachAddList_en.md) API. |
| aclnn invocation  | [test_aclnn_foreach_add_list_v2](examples/test_aclnn_foreach_add_list_v2.cpp) | Invokes the ForeachAddList operator through the [aclnnForeachAddListV2](docs/aclnnForeachAddListV2_en.md) API. |
| Graph mode | -  | Invokes the ForeachAddList operator by constructing the graph through the [operator IR](op_graph/foreach_add_list_proto.h).         |
