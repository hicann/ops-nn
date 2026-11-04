# AddRowRanges

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function

- Operator function: For each row r of the matrix x, sums the row range `[start, end)` of the matrix src specified by indices column by column, and accumulates the result into the corresponding row of x. This operator originates from the `AddRowRanges` operation in the Kaldi speech recognition framework and is mainly used for row range accumulation of feature matrices in speech processing.

- Formula:

$$
x_{out}(r, c) = x(r, c) + \sum_{j=indices[r,0]}^{indices[r,1]-1} src(j, c)
$$

Special handling:

- When `indices[r, 0] >= indices[r, 1]`, the row is an empty range; the operator skips the accumulation and `x_out(r, c) = x(r, c)`
- When `indices[r, 0] == -1` or `indices[r, 1] == -1`, the row is an empty range; the operator skips the accumulation and `x_out(r, c) = x(r, c)`

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>Input matrix, a 2D tensor with shape (M, N), where M is the number of rows and N is the number of columns.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src</td>
      <td>Input</td>
      <td>Source matrix, a 2D tensor with shape (K, N), where K is the number of rows. The number of columns of src must be the same as that of x.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Row range indices, a 2D tensor with shape (M, 2). indices[r, 0] is the start row and indices[r, 1] is the end row (exclusive). A value of -1 indicates an empty range.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td>Output matrix, a 2D tensor whose shape and dtype are the same as those of the input x.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- **Input dimension constraints**: x, src, and indices must be 2D tensors (ND format); other dimensions are not supported.
- **Column count constraint**: the number of columns of src (src.shape[1]) must equal the number of columns of x (x.shape[1]).
- **indices shape constraint**: the shape of indices must be (M, 2), where M equals the number of rows of x (x.shape[0]).
- **dtype constraint**: x and src must use the same dtype, which is float32 or float16. indices must be int32.
- **Empty range handling**: when `indices[r, 0] >= indices[r, 1]` or `indices[r, 0] == -1` or `indices[r, 1] == -1`, the row is an empty range; the operator skips the accumulation, and the output equals the corresponding row of the input x.
- **Index out of bounds**: when indices values exceed the [0, K) range, they are clipped according to numpy slicing semantics: negative indices are resolved to K+value (clipped to 0), and positive indices are clipped to K. After clipping, if start >= end, the range is regarded as empty.
- **Empty tensor handling**: when the number of rows or columns of x is 0, the operator directly returns an empty output without reporting an error.
- **Determinism**: this operator is deterministic computation. Each output element is computed independently by a single thread, without atomicAdd, and the accumulation order is fixed from start to end.
- **Precision description**: in the float16 scenario, float32 intermediate variables are used for accumulation (Kahan compensated summation), and the result is converted back to float16 after the accumulation is complete, avoiding the precision loss of float16 accumulation. In the float32 scenario, Kahan compensated summation is used.
- **NaN/Inf propagation**: if NaN participates in the accumulation, the result is NaN; Inf accumulation follows the IEEE 754 standard (Inf+Inf=Inf, Inf+(-Inf)=NaN).
- **Data range**: the total number of output elements (M × N) must not exceed INT32_MAX (about 2.1 billion).

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>GE graph mode</td>
    <td><a href="./examples/test_geir_add_row_ranges.cpp">test_geir_add_row_ranges</a></td>
    <td>Builds the graph and invokes the AddRowRanges operator through the <a href="./op_graph/add_row_ranges_proto.h">operator IR</a></td>
  </tr>
</tbody>
</table>
