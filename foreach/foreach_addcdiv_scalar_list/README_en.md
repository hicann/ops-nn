# ForeachAddcdivScalarList

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

- Operator function: performs element-wise addition, multiplication, and division on multiple tensors, where $x2_{i}$ and $x3_{i}$ are divided element-wise, the result is multiplied by $scalars_{i}$, and then the product is added to $x1_{i}$.
- Formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}], x3 = [{x3_0}, {x3_1}, ... {x3_{n-1}}]\\
  scalars = [{scalars_0}, {scalars_1}, ... {scalars_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = {x1}_{i}+ \frac{{x2}_{i}}{{x3}_{i}}\times{scalars_i} (i=0,1,...n-1)
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
      <td>Represents the first input tensor list of the addition in the mixed operation, corresponding to `x1` in the formula. All tensors in this parameter have the same data type. Non-contiguous input tensors are automatically converted to contiguous tensors by the framework before computation.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Represents the first input tensor list of the division in the mixed operation, corresponding to `x2` in the formula. Its data type, data format, and shape are consistent with those of input `x1`, and all tensors in this parameter have the same data type. Non-contiguous input tensors are automatically converted to contiguous tensors by the framework before computation.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>Input</td>
      <td>Represents the second input tensor list of the division in the mixed operation, corresponding to `x3` in the formula. Its data type, data format, and shape are consistent with those of input `x1`, and all tensors in this parameter have the same data type. Non-contiguous input tensors are automatically converted to contiguous tensors by the framework before computation.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalars</td>
      <td>Input</td>
      <td>Represents the second input tensor of the multiplication in the mixed operation, corresponding to `scalars` in the formula. The number of elements equals the number of tensors in `x1`. Its data type and data format are consistent with those of input `x1`. Non-contiguous input tensors are automatically converted to contiguous tensors by the framework before computation.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor list of the mixed operation, corresponding to `y` in the formula. The tensor list length is consistent with that of `x1`, its data type and data format are consistent with those of input `x1`, and its shape size (total number of elements) is consistent with that of the corresponding tensor in `x1`. All tensors in this parameter have the same data type. Non-contiguous output tensors are automatically converted to contiguous tensors by the framework before computation.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- A single tensor list contains no more than 256 tensors. If the upper limit is exceeded, the operator execution fails.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_foreach_addcdiv_scalar_list](examples/test_aclnn_foreach_addcdiv_scalar_list.cpp) | Invokes the ForeachAddcdivScalarList operator through the [aclnnForeachAddcdivScalarList](docs/aclnnForeachAddcdivScalarList_en.md) interface. |
| Graph mode | -  | Invokes the ForeachAddcdivScalarList operator by building a graph through the [operator IR](op_graph/foreach_addcdiv_scalar_list_proto.h).         |
