# GatherV2

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |
| <term>KirinX90 processor series products</term>    |    √    |
| <term>Kirin9030 processor series products</term>   |    √    |

## Function

- Operator function: Extracts elements from the specified dimension dim of the input tensor according to the subscript indices in index, and saves them into the out tensor.
- Computation process:
  For example, given the input tensor $x=\begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}$ and the index tensor idx=[1, 0].
  - Result of dim=0: $y=\begin{bmatrix}4 & 5 & 6 \\ 1 & 2 & 3\end{bmatrix}$

  - Result of dim=1: $y=\begin{bmatrix}2 & 1\\ 5 & 4\\ 8 & 7\end{bmatrix}$

  The specific computation process is as follows:
  Taking a 3D tensor as an example, for tensor x with shape (3,2,2), x =$\begin{bmatrix}[[1,&2],&[3,&4]], \\ [[5,&6],&[7,&8]], \\ [[9,&10],&[11,&12]]\end{bmatrix}$   idx=[1, 0];   the subscripts of tensor x corresponding to dim=0,1,2 are $l, m, n$ respectively, and idx is one-dimensional (for the zero-dimensional case, treat it as one-dimensional with size 1).
  - dim is 0: I=index[i];  &nbsp;&nbsp;   y$[i][m][n]$ = x$[I][m][n]$

  - dim is 1: J=index[j];  &nbsp;&nbsp;&nbsp;    y$[l][j][n]$ = x$[l][J][n]$

  - dim is 2: K=index[k]; &nbsp;  y$[l][m][k]$ = x$[l][m][K]$

## Parameters

<table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 215px">
  <col style="width: 163px">
  <col style="width: 287px">
  <col style="width: 439px">
  <col style="width: 135px">
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
        <td>The input tensor, x in the formula.</td>
        <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, COMPLEX64, COMPLEX128</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>Input</td>
        <td>The input subscripts, used to specify the index positions, idx in the formula.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>axis</td>
        <td>Input</td>
        <td>A scalar that represents the axis (dimension) of the operation, dim in the formula.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td>The result of the input, y in the formula.</td>
        <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, COMPLEX64, COMPLEX128</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>batch_dims</td>
        <td>Optional attribute</td>
        <td><ul><li>Specifies the batch dimension.</li><li>The default value is 0.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>negative_index_support</td>
        <td>Optional attribute</td>
        <td><ul><li>Specifies whether negative indices are supported. A value of true means negative indices are supported, and a value of false means negative indices are not supported.</li><li>The default value is false.</li></ul></td>
        <td>BOOL</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_gather_v2](./examples/test_aclnn_gather_v2.cpp) | Calls the GatherV2 operator through the [aclnnGatherV2](./docs/aclnnGatherV2_en.md) interface. |
| Graph-mode invocation | [test_geir_gather_v2](./examples/test_geir_gather_v2.cpp) | Calls the GatherV2 operator by building a graph through the [operator IR](./op_graph/gather_v2_proto.h). |
