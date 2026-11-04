# ForeachAddcdivList

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

- Operator function: Performs element-wise addition, multiplication, and division operations on multiple tensors. $x2_{i}$ and $x3_{i}$ are divided element-wise, the result is multiplied by scalars, and then the product is added to $x1_{i}$.
- Formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}], x3 = [{x3_0}, {x3_1}, ... {x3_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = {x1}_{i}+ \frac{{x2}_{i}}{{x3}_{i}}\times{scalars} (i=0,1,...n-1)
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
      <td>Empty tensors are supported. Represents the first input tensor list of the addition in the mixed operation, which corresponds to `x1` in the formula. All tensors in this parameter have the same data type, and the number of dimensions (Dim) of each tensor is not greater than 8.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Empty tensors are supported. Represents the first input tensor list of the division in the mixed operation, which corresponds to `x2` in the formula. The data type and data format are the same as those of the input parameter `x1`, and the shape size (total number of elements) is the same as that of the tensor at the corresponding position in `x1`. The number of dimensions (Dim) of each tensor is not greater than 8. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>Input</td>
      <td>Empty tensors are supported. Represents the second input tensor list (divisor) of the division in the mixed operation, which corresponds to `x3` in the formula. The data type and data format are the same as those of the input parameter `x1`, and the shape size (total number of elements) is the same as that of the tensor at the corresponding position in `x1`. The number of dimensions (Dim) of each tensor is not greater than 8. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalars</td>
      <td>Input</td>
      <td>Empty tensors are not supported. Represents the second input tensor of the multiplication in the mixed operation, which corresponds to `scalars` in the formula. The number of elements equals the number of tensors in `x1`. The data type and data format are the same as the data type and data format of the input parameter `x1`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Empty tensors are supported. Represents the output tensor list of the mixed operation, which corresponds to `y` in the formula. The data type and data format are the same as the data type and data format of the input parameter `x1`, and the shape size is greater than or equal to the shape size of the input parameter `x1`. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.
- Ascend 950PR&950DT products: The data type of scalars is the same as that of the input parameter `x1`.

## Constraints

- The tensor lists `x1`, `x2`, and `x3` must contain the same number of tensors, which must be consistent with the number of elements of `y` and `scalars`. A single tensor list contains no more than 256 tensors.
- The tensors at corresponding positions in the tensor lists `x1`, `x2`, and `x3` must have the same shape size (total number of elements).

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_addcdiv_list](examples/test_aclnn_foreach_addcdiv_list.cpp) | Calls the ForeachAddcdivList operator through the [aclnnForeachAddcdivList](docs/aclnnForeachAddcdivList_en.md) API. |
