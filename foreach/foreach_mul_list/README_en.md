# ForeachMulList

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

- Operator function: Performs element-wise multiplication on two input tensor lists.

- Formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = x1_i*{x2_i} (i=0,1,...n-1)
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
      <td>First input tensor list of the multiplication, corresponding to `x1` in the formula. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Second input tensor list of the multiplication. Its data type, data format, and shape are consistent with the data type, data format, and shape of the input parameter `x1`. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor list of the multiplication, corresponding to `y` in the formula. Its data type and data format are consistent with the data type and data format of the input parameter `x1`, and its shape size is greater than or equal to the shape size of the input parameter `x1`. All tensors in this parameter have the same data type.</td>
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
| aclnn invocation  | [test_aclnn_foreach_mul_list](examples/test_aclnn_foreach_mul_list.cpp) | Invokes the ForeachMulList operator through the [aclnnForeachMulList](docs/aclnnForeachMulList_en.md) interface. |
