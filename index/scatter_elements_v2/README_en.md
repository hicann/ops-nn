# ScatterElementsV2

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

- Operator function: replaces, accumulates, or accumulates by multiplication the values in the tensor src into the tensor self one by one according to the specified axis, direction, and corresponding position relationship.

- Example:
  For a 3D tensor, self is updated according to the following rules:

  ```bash
  self[index[i][j][k]][j][k] += src[i][j][k] # if dim == 0 && reduction == 1
  self[i][index[i][j][k]][k] *= src[i][j][k] # if dim == 1 && reduction == 2
  self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2 && reduction == 0
  ```

  The following requirements must be met during computation:
  - The numbers of dimensions of self, index, and src must be the same.
  - For each dimension d, the restriction index.size(d) <= src.size(d) applies.
  - For each dimension d, if d != dim, the restriction index.size(d) <= self.size(d) applies.
  - The value of dim must be in the range [-the number of dimensions of self, the number of dimensions of self - 1].
  - The number of dimensions of self should be less than or equal to 8.
  - The values of index along the dimension dim must be in the range [0, self.size(dim)-1].

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
      <td>self</td>
      <td>Input</td>
      <td>`self` in the formula, an aclTensor on the Device.</td>
      <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Input</td>
      <td>The dimension for scattering. The data type is INT64.</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Input</td>
      <td>`index` in the formula, an aclTensor on the Device.</td>
      <td>INT32, INT64.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src</td>
      <td>Input</td>
      <td>`src` in the formula, an aclTensor on the Device.</td>
      <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Input</td>
      <td>A string on the Host, selecting the reduction operation to apply.</td>
      <td>string</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The output in the formula.</td>
      <td>FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_scatter](./examples/test_aclnn_scatter.cpp) | Invokes the ScatterElementsV2 operator through the [aclnnScatter&aclnnInplaceScatter](./docs/aclnnScatter&aclnnInplaceScatter_en.md) API. |
