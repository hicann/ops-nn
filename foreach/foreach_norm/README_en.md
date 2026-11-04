# ForeachNorm

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

- Operator function: Performs the norm operation on each tensor in the input tensor list.
- Formula:
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:

    $$
    x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
    $$

    $$
    y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
    $$

    where $x_i$ denotes the $i$-th tensor in the input list, $m_i$ denotes the number of elements in $x_i$, and $y_i$ denotes the norm result corresponding to $x_i$:

    $$
    y_i = \left(\sum_{j=0}^{m_i-1}|x_{i,j}|^{p}\right)^{\frac{1}{{p}}}
    $$

  - <term>Ascend 950PR&950DT products</term>:

    $$
    x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
    $$

    $$
    y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
    $$

    where $x_i$ denotes the $i$-th tensor in the input list, $m_i$ denotes the number of elements in $x_i$, and $y_i$ denotes the norm result corresponding to $x_i$:

    - scalar = +Inf:

      $$
      y_i = \max_{0 \le j < m_i}|x_{i,j}|
      $$

    - int(scalar) = 1:

      $$
      y_i = \sum_{j=0}^{m_i-1}|x_{i,j}|
      $$

    - other:

      $$
      y_i = \left(\sum_{j=0}^{m_i-1}|x_{i,j}|^{2}\right)^{\frac{1}{{2}}}
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
      <td>Represents the input tensor list of the norm operation, corresponding to `x` in the formulas. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalar</td>
      <td>Input</td>
      <td>Represents the order of the norm operation, corresponding to `p` in the formulas.</td>
      <td>FLOAT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor list of the norm operation, corresponding to `y` in the formulas. The data type and data format are consistent with those of the input `x`, and the shape size is 1.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Ascend 950PR&950DT products</term>:
  - The maximum length supported by the input `x` and the output `y` is 256.
  - For the value of the input `scalar`, only 1 or 2 is supported.

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: the data types of x and y do not support BFLOAT16.

## Constraints

- The output does not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_foreach_norm](examples/test_aclnn_foreach_norm.cpp) | Invokes the ForeachNorm operator through the [aclnnForeachNorm](docs/aclnnForeachNorm_en.md) interface. |
| Graph-mode invocation | -  | Builds the graph and invokes the ForeachNorm operator through the [operator IR](op_graph/foreach_norm_proto.h).         |
