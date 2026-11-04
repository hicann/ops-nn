# ForeachLerpScalar

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

- Operator function: performs interpolation computation on the elements at corresponding positions of two tensor lists, where the scalar weight is the interpolation coefficient.
- Formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = x1_i + {\rm weight} × (x2_i - x1_i) (i=0,1,...n-1)
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
      <td>The first input tensor list for the linear interpolation computation, corresponding to `x1` in the formula. The data type, data format, and shape are consistent with those of the input parameter `x2`. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The second input tensor list for the linear interpolation computation, corresponding to `x2` in the formula. The data type, data format, and shape are consistent with the data type, data format, and shape of the input parameter `x1`. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>The interpolation coefficient for the linear interpolation computation, corresponding to `weight` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output tensor list for the linear interpolation computation, corresponding to `y` in the formula. The data type and data format are consistent with the data type and data format of the input parameter `x1`, and the shape size is greater than or equal to the shape size of the input parameter `x1`. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- For <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, the shape of `x1` is consistent with those of the input parameter `x2` and the output parameter `y`.
- For <term>Ascend 950PR&950DT products</term>, the shape size of `x1` is consistent with the shape sizes of the input parameter `x2` and the output parameter `y`. The maximum number of contained tensors is 50.

- For <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>, the data types of x1, x2, and y do not support BFLOAT16.

## Constraints

- The output does not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_lerp_scalar](examples/test_aclnn_foreach_lerp_scalar.cpp) | Invokes the ForeachLerpScalar operator through the [aclnnForeachLerpScalar](docs/aclnnForeachLerpScalar_en.md) API. |
| Graph mode | -  | Invokes the ForeachLerpScalar operator through graph construction using the [operator IR](op_graph/foreach_lerp_scalar_proto.h).         |
