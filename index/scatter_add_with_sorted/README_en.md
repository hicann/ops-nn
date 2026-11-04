# ScatterAddWithSorted

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
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: replaces/accumulates/accumulates multiplicatively, one by one, the values in the tensor src into the tensor self according to the specified axis, direction, and corresponding positional relationship.

- Sample:
  For a 3D tensor, self is updated according to the following rules:

  ```cpp
  self[index[i][j][k]][j][k] += src[i][j][k] # if dim == 0 && reduction == 1
  self[i][index[i][j][k]][k] *= src[i][j][k] # if dim == 1 && reduction == 2
  self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2 && reduction == 0
  ```

  The computation must satisfy the following requirements:
  - The numbers of dimensions of self, index, and src must be the same.
  - For each dimension d, the restriction index.size(d) <= src.size(d) holds.
  - For each dimension d, if d != dim, the restriction index.size(d) <= self.size(d) holds.
  - The value of dim must be in the range [-the number of dimensions of self, the number of dimensions of self - 1].
  - The number of dimensions of self should be less than or equal to 8.
  - The value of the dimension dim in index must be in the range [0, self.size(dim)-1].

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
      <td>var</td>
      <td>Input</td>
      <td>`self` in the formula, an aclTensor on the Device side.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>`index` in the formula, an aclTensor on the Device side.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorted_index</td>
      <td>Input</td>
      <td>`src` in the formula, an aclTensor on the Device side.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pos</td>
      <td>Input</td>
      <td>`src` in the formula, an aclTensor on the Device side.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>`self` in the formula, the output in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraints

None
