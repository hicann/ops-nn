# ScatterElements

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √  |
| <term>Atlas A3 products</term>    |    √    |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |

## Function

- Operator function: Replaces/accumulates/accumulates multiplicatively, one by one, the values in the tensor updates into the output tensor according to the specified axis and direction and the corresponding position relationship, and the values at the non-updated positions of the output tensor are the same as those of the input tensor.

- Example:
  For a 3D tensor, the output is updated according to the following rules:

  ```bash
  y[indices[i][j][k]][j][k] += updates[i][j][k] # if dim == 0 && reduction == 1
  y[i][indices[i][j][k]][k] *= updates[i][j][k] # if dim == 1 && reduction == 2
  y[i][j][indices[i][j][k]] = updates[i][j][k]  # if dim == 2 && reduction == 0
  ```

  The following requirements must be met during computation:
  - The numbers of dimensions of data, indices, and updates must be the same.
  - For each dimension d, the constraint indices.size(d) <= updates.size(d) must be met.
  - For each dimension d, if d != dim, the constraint indices.size(d) <= data.size(d) must be met.
  - The value of dim must be in [-number of dimensions of data, number of dimensions of data-1].
  - The number of dimensions of data should be less than or equal to 8.
  - The values of indices must be in [0, data.size(dim)-1].

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>data</td>
      <td>Input</td>
      <td>`data` in the formula, an aclTensor on the Device side.</td>
      <td>BFLOAT16, BOOL, COMPLEX128, COMPLEX64, DOUBLE, FLOAT16, FLOAT32, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Input</td>
      <td>Dimension used for scatter, of the data type INT64.</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>`indices` in the formula, an aclTensor on the Device side.</td>
      <td>INT32, INT64.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>`updates` in the formula, an aclTensor on the Device side.</td>
      <td>BFLOAT16, BOOL, COMPLEX128, COMPLEX64, DOUBLE, FLOAT16, FLOAT32, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Input</td>
      <td>String on the Host side, selecting the reduction operation to be applied.</td>
      <td>string</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output in the formula.</td>
      <td>BFLOAT16, BOOL, COMPLEX128, COMPLEX64, DOUBLE, FLOAT16, FLOAT32, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_scatter_elements](./examples/test_geir_scatter_elements.cpp) | Calls the ScatterElements operator by constructing a graph through the [operator IR](./op_graph/scatter_elements_proto.h). |
| aclnn invocation | [aclnnMaxUnpool2d](./docs/aclnnMaxUnpool2d_en.md) | Calls the L0 API of the ScatterElements operator through the aclnnMaxUnpool2d API, which is actually routed to the ScatterElementsV2 operator on Ascend 950. |
| aclnn invocation | [aclnnMaxUnpool3d](./docs/aclnnMaxUnpool3d_en.md) | Calls the L0 API of the ScatterElements operator through the aclnnMaxUnpool3d API, which is actually routed to the ScatterElementsV2 operator on Ascend 950. |
