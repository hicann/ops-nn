# ApplyCamePart3

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

- **Operator function**: Computes the first-moment update value of the third stage of the CAME (Confidence-guided Adaptive Memory Efficient) optimizer, together with the row, column, and global reduction results.

- **Formula**:

  For the 2D input tensors `u` and `m`, and the scalar inputs `eps`, `beta1`, `clip_threshold`, and `sum_square_u`, let the global number of rows be `global_n` and the global number of columns be `global_m`. First compute:

  $$
  s = \max\left(1,\frac{sum\_square\_u}{global\_n \times global\_m \times clip\_threshold}\right)
  $$

  The implementation follows the conditional branch semantics of A2: the scaled value is used only when it is greater than 1, and 1 is used in other cases.

  $$
  m\_update_{i,j} = (1 - beta1) \times \frac{u_{i,j}}{s} + beta1 \times m_{i,j}
  $$

  When `use_first_moment` is `true`, the output `m` is `m_update`; when it is `false`, the output `m` keeps the input `m`.

  Then let:

  $$
  x_{i,j}=\left(\frac{u_{i,j}}{s}-m\_update_{i,j}\right)^2+eps
  $$

  Perform the row, column, and global reductions respectively:

  $$
  \begin{aligned}
  sum\_u\_r_i &= \sum_j x_{i,j} \\
  sum\_u\_c_j &= \sum_i x_{i,j} \\
  sum\_u\_rc &= \sum_i \sum_j x_{i,j}
  \end{aligned}
  $$

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
    <th>Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data type</th>
    <th>Data format</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>u</td>
    <td>Input</td>
    <td>2D input tensor, which is <code>u</code> in the formula.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>m</td>
    <td>Input</td>
    <td>First-moment input tensor, which must have the same shape as <code>u</code> and is <code>m</code> in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>eps</td>
    <td>Input</td>
    <td>Numerical stability term, a scalar or a one-element 1D tensor.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta1</td>
    <td>Input</td>
    <td>First-moment decay coefficient, a scalar or a one-element 1D tensor.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>clip_threshold</td>
    <td>Input</td>
    <td>Clip threshold, a scalar or a one-element 1D tensor.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sum_square_u</td>
    <td>Input</td>
    <td>Sum of squares of <code>u</code> or the corresponding global statistic, a scalar or a one-element 1D tensor.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>global_shape</td>
    <td>Optional input</td>
    <td>Global 2D shape, which contains the global number of rows and columns.</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_first_moment</td>
    <td>Attribute</td>
    <td>Specifies whether to output the updated first moment. Default: <code>false</code>.</td>
    <td>Bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>m</td>
    <td>Output</td>
    <td>First-moment output tensor, with the same shape and data type as the input <code>m</code>.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sum_u_r</td>
    <td>Output</td>
    <td>Row result after reduction along columns, with the shape <code>[N]</code>.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sum_u_c</td>
    <td>Output</td>
    <td>Column result after reduction along rows, with the shape <code>[M]</code>.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sum_u_rc</td>
    <td>Output</td>
    <td>Global reduction result, with the shape <code>[1]</code>.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraints

- `u` and the input `m` must be 2D ND tensors with the same shape, and both dimensions must be greater than 0. The total number of bytes of the tensors must be within the 64-bit address range.
- `eps`, `beta1`, `clip_threshold`, and `sum_square_u` must be FLOAT32 ND tensors, each containing only one element.
- `global_shape` is an optional INT64 input and must be a 1D tensor of length 2, `[global_n, global_m]`. When it is not provided, the 2D shape of the input `u` is used for the reduction computation.

## Invocation Description

| Invocation method | Invocation sample | Description |
|---|---|---|
| Graph-mode invocation | [test_geir_apply_came_part3.cpp](./examples/test_geir_apply_came_part3.cpp) | Invokes the ApplyCamePart3 operator through graph construction with the [operator IR](./op_graph/apply_came_part3_proto.h). |
