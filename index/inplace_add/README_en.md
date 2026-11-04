# InplaceAdd

English | [简体中文](./README.md)

## Applicable Products

| Product                                                     | Support |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    √     |
| <term>Atlas A3 products</term>|    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                  |    √     |
| <term>Atlas inference products</term>                          |    √     |
| <term>Atlas training products</term>                          |    √     |

## Function

- Operator function: adds the slices in `v` to the corresponding slices of `x` at the positions along dimension 0 specified by `indices`, and outputs the updated result to `y`. The interface does not declare that `x` and `y` share storage.
- Use case: it is compatible with the `InplaceAdd` graph operator of TensorFlow and is used to perform indexed updates by row; this operator is not a common element-wise or broadcast Add.
- Formula: let the shape of `x` be $[N, d_1, \dots, d_{D-1}]$, the shape of `indices` be $[K]$, and the shape of `v` be $[K, d_1, \dots, d_{D-1}]$. Denote $row_j$ as the actual target row corresponding to the j-th update; its determination rule is given in "Constraints". Provided that the target rows do not repeat each other, the formula is as follows:

  $$
  y_{i} =
  \begin{cases}
  x_{i} + v_{j} & \exists j,\ row_j = i \\
  x_{i} & otherwise
  \end{cases}
  $$

  Here $i$ iterates over dimension 0, and $x_i$, $y_i$, and $v_j$ each denote the entire trailing slice at the corresponding subscript. If multiple indices correspond to the same target row, the output content is undefined.

- Example:

  ```text
  x       = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
  indices = [0, 2]
  v       = [[1, 1, 1],
             [2, 2, 2]]

  y       = [[2, 3, 4],
             [4, 5, 6],
             [9, 10, 11]]
  ```

## Parameters

<table style="table-layout: fixed; width: 1460px"><colgroup>
  <col style="width: 90px">
  <col style="width: 140px">
  <col style="width: 280px">
  <col style="width: 350px">
  <col style="width: 480px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>shape/Supported dimensions</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input <code>x</code> in the formula, serving as the update base value.</td>
      <td>[N, d1, ..., d(D-1)], supporting 1 to 8 dimensions.</td>
      <td>FLOAT16, FLOAT, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Input <code>indices</code> in the formula, specifying the positions along dimension 0 of <code>x</code> that need to be updated.</td>
      <td>[K], supporting only 1 dimension.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v</td>
      <td>Input</td>
      <td>Input <code>v</code> in the formula, representing the update values added to the target rows; its data type must be the same as that of <code>x</code>.</td>
      <td>[K, d1, ..., d(D-1)], supporting 1 to 8 dimensions, and the number of dimensions must be the same as that of x.</td>
      <td>FLOAT16, FLOAT, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output <code>y</code> in the formula, with the same shape and data type as <code>x</code>; the interface does not declare storage sharing with <code>x</code>.</td>
      <td>Same as x, supporting 1 to 8 dimensions.</td>
      <td>FLOAT16, FLOAT, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX32, COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- The inputs and outputs of all supported products use the ND format; `x`, `v`, and `y` support 1 to 8 dimensions, and `indices` supports only 1 dimension.
- For products that support dynamic shape/rank, `-1` denotes an unknown dimension and `[-2]` denotes an unknown rank at graph compilation time; the output `y` inherits the shape of `x`. During execution, resolve them into concrete non-negative dimensions first, and then perform validation according to the rank, empty tensor, and scale limits of the corresponding product.
- `v.shape[0]` must equal `indices.shape[0]`; `v.shape[1:]` must equal `x.shape[1:]`.
- The data types of `x`, `v`, and `y` must be consistent, and the data type of `indices` must be INT32.
- When multiple indices correspond to the same target row, the output content is undefined, and the accumulation order, determinism, or specific values are not guaranteed.
- Integer addition does not provide saturation operation guarantees, and callers should not rely on the results after integer overflow.
- This operator does not provide an aclnn single-operator interface and supports only GE graph-mode invocation. The existing same-named `aclnnInplaceAdd` performs the element-wise or broadcast computation `self += alpha * other`, which differs from the indexed row update semantics of this operator, so the two cannot be mixed.
- <term>Ascend 950PR&950DT products</term>:
  - Data types: `x`, `v`, and `y` support FLOAT16, FLOAT, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX32, and COMPLEX64.
  - Input scenario: `indices` supports runtime input or compile-time constant.
  - Shape capability: static shapes, dynamic shapes, and dynamic ranks are supported, with ND input -> ND output.
  - Index rule: when $N>0$, the target row is determined by $row_j=((indices_j \bmod N)+N)\bmod N$.
  - Empty tensors: $K=0$ is supported; when $N=0$, $K=0$ is required, and when any trailing dimension is 0, the output is empty.
  - Scale limits: $N$ and $K$ do not exceed the maximum value of INT32; the trailing dimension sizes, tensor element counts, total processing scale, and byte sizes must be representable in INT64.
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas 200I/500 A2 inference products</term>, and <term>Atlas inference products</term>:
  - Data types: `x`, `v`, and `y` support FLOAT16, FLOAT, and INT32.
  - Input scenario: `indices` supports runtime input or compile-time constant.
  - Shape capability: static shapes, dynamic shapes, and dynamic ranks are supported, with ND input -> ND output.
  - Index rule: $0\leq indices_j<N$ is required; empty tensors are not supported.
- <term>Atlas training products</term>:
  - When `x`, `v`, and `y` are FLOAT or INT32, `indices` supports runtime input or compile-time constant, static shapes, dynamic shapes, and dynamic ranks are supported, and ND input -> ND output applies.
  - When `x`, `v`, and `y` are FLOAT16, `indices` must be a compile-time constant, only static shapes are supported, and ND input -> ND output applies.
  - Index rule: $0\leq indices_j<N$ is required; empty tensors are not supported.

## Invocation Description

| Invocation method | Sample code | Description |
| :------- | :------- | :--- |
| GE graph mode | [test_geir_inplace_add](examples/test_geir_inplace_add.cpp) | Invokes the InplaceAdd operator by building a graph through the [operator IR](op_graph/inplace_add_proto.h), and supports FLOAT, FLOAT16, BFLOAT16, INT8, UINT8, and INT32 inputs. |
