# ForeachErfc

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

- Operator function: performs the complementary error function operation, which integrates from x to infinity, element by element.
- Computation formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = {\rm erfc}(x_i) = 1 - {\rm erf}(x_i) = 1 - \frac{2}{\sqrt{\pi}} \int_{0}^{x_i} e^{-t^2} dt = \frac{2}{\sqrt{\pi}} \int_{x_i}^{\infty} e^{-t^2} dt   (i=0,1,...n-1)
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
      <td>x</td>
      <td>Input</td>
      <td>Denotes the input tensor list for the complementary error function operation, corresponding to `x` in the formula. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the output tensor list for the complementary error function operation, corresponding to `y` in the formula. All tensors in this parameter have the same data type. Its data type and data format are consistent with those of input parameter `x`, and its shape size is greater than or equal to the shape size of input parameter `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- The numbers of tensors in x and out must be the same, and the shape size of each tensor in out must be greater than or equal to that of the tensor at the corresponding position in x.
- All tensors within a single tensor list of x or out must have the same data type, and the data types of x and out must be consistent, all within the supported range declared in the parameter table.
- The number of dimensions of a single tensor cannot exceed 8.
- The number of tensors in a single tensor list of x/y cannot exceed 256.
- When the numbers of tensors in x and out are 0 (empty lists), the API returns success directly without performing computation.
- Non-contiguous tensors are supported: non-contiguous inputs/outputs are automatically made contiguous for computation, and results are written back in the original layout.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_foreach_erfc](examples/test_aclnn_foreach_erfc.cpp) | Calls the ForeachErfc operator through the [aclnnForeachErfc](docs/aclnnForeachErfc_en.md) interface. |
| Graph mode | -  | Calls the ForeachErfc operator by constructing a graph through the [operator IR](op_graph/foreach_erfc_proto.h).         |
