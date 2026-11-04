# ForeachSubList

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

- Operator function: performs element-wise subtraction on the two input tensor lists. You can adjust the subtraction coefficient through the alpha parameter.

- Formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}]\\
  x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = x1_i-{x2_i}*alpha (i=0,1,...n-1)
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
      <td>The first input for the subtraction operation, corresponding to `x1` in the formula. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The second input for the subtraction operation, corresponding to `x2` in the formula. The data types of all tensors in this parameter are consistent. The data type, data format, and shape are consistent with the data type, data format, and shape of the input parameter `x1`.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>The coefficient of the second input for the subtraction operation, corresponding to `alpha` in the formula. The number of elements is 1.</td>
      <td>FLOAT32, FLOAT16, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output result of the subtraction operation, corresponding to `y` in the formula. The data types of all tensors in this parameter are consistent. The data type and data format are consistent with the data type and data format of the input parameter `x1`, and the shape size is greater than or equal to the shape size of the input parameter `x1`.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- For <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>, BFLOAT16, INT16, INT8, and UINT8 are not supported.
- The output does not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_sub_list](examples/test_aclnn_foreach_sub_list.cpp) | Invokes the ForeachSubList operator through the [aclnnForeachSubList](docs/aclnnForeachSubList_en.md) API. |
| aclnn API  | [test_aclnn_foreach_sub_list_v2](examples/test_aclnn_foreach_sub_list_v2.cpp) | Invokes the ForeachSubList operator through the [aclnnForeachSubListV2](docs/aclnnForeachSubListV2_en.md) API. |
