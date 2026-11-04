# ApplyCamePart1

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term> | √ |
|  <term>Atlas A3 products</term>| √ |
|  <term>Atlas A2 products</term> | √ |
|  <term>Atlas 200I/500 A2 inference products</term> | × |
|  <term>Atlas inference products</term> | × |
|  <term>Atlas training products</term> | × |

## Function

- Operator function: computes the squared gradient reduction values of the first stage of the CAME (Confidence-guided Adaptive Memory Efficient) optimizer.

- Computation formula:

  For the elements $g_{b,\dots,i,j}$ on the last two dimensions $[N, M]$ of input `grad`, first compute:

  $$
  x_{b,\dots,i,j} = g_{b,\dots,i,j}^{2} + eps
  $$

  Then perform the row, column, and 2D reductions separately for each batch slice:

  $$
  \begin{aligned}
  sum\_grad\_r_{b,\dots,i} &= \sum_j x_{b,\dots,i,j} \\
  sum\_grad\_c_{b,\dots,j} &= \sum_i x_{b,\dots,i,j} \\
  sum\_grad\_rc_{b,\dots} &= \sum_i \sum_j x_{b,\dots,i,j}
  \end{aligned}
  $$

  where $b,\dots$ denote the batch dimension indexes, and $i$ and $j$ denote the indexes of the last two dimensions respectively.

## Parameters

<table style="table-layout: fixed; width: 100%">
<colgroup>
<col style="width: 16%">
<col style="width: 18%">
<col style="width: 38%">
<col style="width: 16%">
<col style="width: 12%">
</colgroup>
<thead>
  <tr>
    <th>Parameter name</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Data type</th>
    <th>Data format</th>
  </tr>
</thead>
<tbody>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Gradient tensor, whose last two dimensions are <code>[N, M]</code>. For the specific shape requirements, see Constraints.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Input</td>
      <td>Numerical stability constant. For the specific shape requirements, see Constraints.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_grad_r</td>
      <td>Output</td>
      <td>Result obtained by reducing over the last dimension. For the specific shape, see Constraints.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  <tr>
    <td>sum_grad_c</td>
    <td>Output</td>
    <td>Result obtained by reducing over the second-to-last dimension. For the specific shape, see Constraints.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
    <tr>
      <td>sum_grad_rc</td>
      <td>Output</td>
      <td>Result obtained by reducing over the last two dimensions. For the specific shape, see Constraints.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- `grad` must be a non-empty ND tensor, and the size of all dimensions must be greater than 0.
- `eps` must be an ND tensor containing only one FLOAT element.
- <term>Ascend 950PR&950DT products</term>:
  - The rank of `grad` is no less than 2, the last two dimensions are `[N, M]`, and the remaining dimensions are batch dimensions.
  - The product of the sizes of all dimensions of `grad` cannot exceed the maximum value of INT64.
  - `eps` supports any ND tensor containing only one element.
  - The shapes of `sum_grad_r`, `sum_grad_c`, and `sum_grad_rc` are `[batch..., N]`, `[batch..., M]`, and `[batch...]` respectively; when the rank of `grad` is 2, `sum_grad_rc` is a scalar.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - `grad` supports only the 2D tensor with the shape `[N, M]`.
  - `eps` supports only scalars.
  - The shapes of `sum_grad_r`, `sum_grad_c`, and `sum_grad_rc` are `[N]`, `[M]`, and `[1]` respectively.
- This operator supports only the GE graph mode invocation and does not provide a public aclnn interface.

## Invocation Description

| Invocation method | Invocation sample | Description |
|---|---|---|
| Graph-mode invocation | [test_geir_apply_came_part1.cpp](./examples/test_geir_apply_came_part1.cpp) | Calls the ApplyCamePart1 operator by constructing a graph through the [operator IR](./op_graph/apply_came_part1_proto.h). |
