# InplaceSub

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

- Operator function: subtracts the values in v from the corresponding slices of x according to the 0th-dimension positions specified by indices, and returns the result through output y.

- Computation logic:

  The initial value of y is the same as x. For each position i in indices, the following update is performed:

  ```text
  y[indices[i], ...] = y[indices[i], ...] - v[i, ...]
  ```

- Example:

  Input x is:

  &emsp;&emsp;[[1, 2, 3],

  &emsp;&emsp;&nbsp;[4, 5, 6],

  &emsp;&emsp;&nbsp;[7, 8, 9]]

  indices = [0, 2], and v is:

  &emsp;&emsp;[[1, 1, 1],

  &emsp;&emsp;&nbsp;[2, 2, 2]]

  The computation result of the operator is:

  &emsp;&emsp;[[0, 1, 2],

  &emsp;&emsp;&nbsp;[4, 5, 6],

  &emsp;&emsp;&nbsp;[5, 6, 7]]

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
      <td>Input tensor to be updated. Output y has the same shape as x, in ND format.</td>
      <td>COMPLEX64, FLOAT16, FLOAT, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>One-dimensional index tensor in ND format, specifying the positions of the 0th dimension on the left side of x.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v</td>
      <td>Input</td>
      <td>Update value tensor. Its data type is the same as x, the size of its 0th dimension equals the length of indices, and its other dimensions are the same as x, in ND format.</td>
      <td>COMPLEX64, FLOAT16, FLOAT, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Updated tensor, with the same shape and data type as x, in ND format.</td>
      <td>COMPLEX64, FLOAT16, FLOAT, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The data types of x, v, and y must be consistent.
- The number of dimensions of x must be greater than 0, and indices must be a one-dimensional tensor.
- v.shape[0] must equal the length of indices, and v.shape[1:] must equal x.shape[1:].
- When x.shape[0] is 0, the length of indices must be 0; when any trailing dimension is 0, the output is an empty tensor.
- indices are normalized by taking the modulo of the 0th-dimension size of x.
- When duplicate values exist in indices, the output result is undefined.
- <term>Ascend 950PR&950DT products</term>:
  - `x.shape[0]` and `indices.shape[0]` must not exceed the maximum value of INT32 (2147483647), and the number of elements of the trailing dimension and the number of tensor elements must be representable within INT64. For COMPLEX32 and COMPLEX64, the number of elements is counted after expansion into the real and imaginary parts.
  - During graph compilation, `-1` indicates an unknown dimension and `[-2]` indicates an unknown rank. Output `y` inherits the shape of `x`. During execution, tiling verification is performed with the specific non-negative dimensions.
  - Integer subtraction wraps around according to the corresponding bit width, and saturation truncation is not performed.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE graph mode | [test_geir_inplace_sub](./examples/test_geir_inplace_sub.cpp) | Invokes the InplaceSub operator through the [operator IR](./op_graph/inplace_sub_proto.h) graph construction method. |
