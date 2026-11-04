# TensorScatterUpdate

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function

- Operator function: a sparse update operator that updates the specified elements or slices of a tensor according to indices. It accepts an input tensor, an index tensor, and an update value tensor, and outputs the updated tensor. It is commonly used for partial modification, sparse gradient updates, or sparse data operations.
- Formula:
  Assume that the input tensor is $X$, the index set is $I=\{i_k\}$, the update values are $U=\{u_k\}$, and the output tensor is $Y$. For any position $j$:

  $$
  Y_j =
  \begin{cases}
  u_k, & \text{if } j = i_k \\
  X_j, & \text{otherwise}
  \end{cases}
  $$

  When the position $j$ is in the index set, the update value replaces the original value; otherwise, the value of the original tensor is kept unchanged.
  In this formula, $k$ denotes the $k$-th update item, $i_k$ denotes the update position corresponding to that update item, and $u_k$ denotes the update value corresponding to that update item.

## Parameters

<table style="undefined;table-layout: fixed; width: 1350px"><colgroup>
  <col style="width: 120px">
  <col style="width: 180px">
  <col style="width: 340px">
  <col style="width: 560px">
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
      <td>X in the formula, the input tensor to be updated, which provides the baseline data before the update.</td>
      <td>FLOAT16, FLOAT, DOUBLE, INT64, INT32, UINT8, UINT16, UINT32, UINT64, INT8, INT16, BOOL, COMPLEX64, COMPLEX128, QINT8, QUINT8, QINT16, QUINT16, QINT32, BFLOAT16, STRING</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>The index set I in the formula, used to specify the positions in the input tensor X that need to be updated.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>The update value set U in the formula, which denotes the update values written to the positions corresponding to the index set I. The data type is the same as x, and the shape is the same as indices.</td>
      <td>FLOAT16, FLOAT, DOUBLE, INT64, INT32, UINT8, UINT16, UINT32, UINT64, INT8, INT16, BOOL, COMPLEX64, COMPLEX128, QINT8, QUINT8, QINT16, QUINT16, QINT32, BFLOAT16, STRING</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Y in the formula, which denotes the updated output tensor. The data type is the same as x, and the shape is the same as x.</td>
      <td>FLOAT16, FLOAT, DOUBLE, INT64, INT32, UINT8, UINT16, UINT32, UINT64, INT8, INT16, BOOL, COMPLEX64, COMPLEX128, QINT8, QUINT8, QINT16, QUINT16, QINT32, BFLOAT16, STRING</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_tensor_scatter_update](./examples/test_geir_tensor_scatter_update.cpp) | Invokes the TensorScatterUpdate operator by constructing the graph through the [operator IR](./op_graph/tensor_scatter_update_proto.h). |
