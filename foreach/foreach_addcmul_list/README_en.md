# ForeachAddcmulList

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

- Operator function: first performs element-wise multiplication between tensor list x2 and tensor list x3, multiplies the result by tensor scalars, and finally performs element-wise addition between the previously computed result and tensor list x1.
- Formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}], x3 = [{x3_0}, {x3_1}, ... {x3_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = x1_{i} + {\rm scalars} × x2_{i} × x3_{i} (i=0,1,...n-1)
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
      <td>Empty tensors are supported. Denotes the first input tensor list of the addition in the mixed operation, corresponding to `x1` in the formula. The data types of all tensors in this parameter are consistent, and the number of dimensions (Dim) of each tensor is not greater than 8.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Empty tensors are supported. Denotes the second input tensor list of the multiplication in the mixed operation, corresponding to `x2` in the formula. The data type and data format are consistent with input parameter `x1`, and the shape size (total number of elements) is consistent with the tensor at the corresponding position of `x1`; the number of dimensions (Dim) of each tensor is not greater than 8. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>Input</td>
      <td>Empty tensors are supported. Denotes the third input tensor list of the multiplication in the mixed operation, corresponding to `x3` in the formula. The data type and data format are consistent with input parameter `x1`, and the shape size (total number of elements) is consistent with the tensor at the corresponding position of `x1`; the number of dimensions (Dim) of each tensor is not greater than 8. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalars</td>
      <td>Input</td>
      <td>Empty tensors are not supported. Denotes the first input tensor of the multiplication in the mixed operation, corresponding to `scalars` in the formula. The number of elements equals the number of tensors in `x1`. The data type and data format are consistent with the data type and data format of input parameter `x1`.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Empty tensors are supported. Denotes the output tensor list of the mixed operation, corresponding to `y` in the formula. The data type and data format are consistent with the data type and data format of input parameter `x1`, and the shape size is greater than or equal to the shape size of input parameter `x1`. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.
- Ascend 950PR&950DT products: the data type of scalars is consistent with input parameter `x1` (when `x1` is FLOAT32/FLOAT16/BFLOAT16/INT32, scalars is of the corresponding same type).

## Constraints

- The tensor lists `x1`, `x2`, and `x3` must contain the same number of tensors, which must be consistent with the numbers of elements of `y` and `scalars`, and the number of tensors in a single tensor list does not exceed 256.
- The shape sizes (total numbers of elements) of the tensors at corresponding positions in tensor lists `x1`, `x2`, and `x3` must be equal.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_foreach_addcmul_list](examples/test_aclnn_foreach_addcmul_list.cpp) | Invokes the ForeachAddcmulList operator through the [aclnnForeachAddcmulList](docs/aclnnForeachAddcmulList_en.md) interface. |
