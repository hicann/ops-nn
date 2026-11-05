# ScatterNdUpdate

English | [简体中文](./README.md)


## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |   √   |
| <term>Atlas A3 products</term>    |    √    |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |
| <term>KirinX90 processor series products</term>                             |    √     |
| <term>Kirin9030 processor series products</term>                              |    √   |

## Function

- Operator function: according to indices, the operator applies updates to individual values or slices of a given variable.
- Formula: ref[indices[k]]=updates[k]
    - ref is a tensor of dimension P;
    - indices is an integer tensor of dimension Q;
    - The shape of indices must be [d0,..., d_Q-2, K], where 0<K<=P;
    - updates is a tensor of dimension Q−1+P−K: [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]]

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
      <td>Denotes a tensor to be updated, equivalent to `ref` in the formula</td>
      <td>DT_INT8, DT_INT64, DT_BOOL, DT_FLOAT16, DT_BF16, DT_FLOAT, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>An index tensor, indexing into `ref` in the formula</td>
      <td>INT32, INT64.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>This tensor is used to update the var tensor, and it must have the same data type as the input var tensor</td>
      <td>DT_INT8, DT_INT64, DT_BOOL, DT_FLOAT16, DT_BF16, DT_FLOAT, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>Denotes the updated tensor</td>
      <td>DT_INT8, DT_INT64, DT_BOOL, DT_FLOAT16, DT_BF16, DT_FLOAT, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>Optional attribute</td>
      <td>Optional attribute, with the default value `false`. If the value is `true`, this operation is protected by a lock</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>
  <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: the FLOAT8_E4M3FN, FLOAT8_E5M2, and HIFLOAT8 data types are not supported on these models.
  <term>KirinX90 processor series products</term> and <term>Kirin9030 processor series products</term>: the DT_INT64, DT_BF16, FLOAT8_E4M3FN, FLOAT8_E5M2, and HIFLOAT8 data types are not supported on these models, and INT64 is not supported for indices.

## Constraints

- Input shape limits:
  - indices must have at least 2 dimensions, and the size of its last dimension cannot exceed the dimension size of varRef.
  - Suppose the size of the last dimension of indices is a; then the shape of updates equals the shape of indices excluding the last dimension plus the shape of varRef excluding the first a dimensions. Example: if the shape of varRef is (4, 5, 6) and the shape of indices is (3, 2), the shape of updates must be (3, 6).

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface | [test_aclnn_scatter_nd_update](examples/test_aclnn_scatter_nd_update.cpp) | Calls the ScatterNdUpdate operator through the [aclnnScatterNdUpdate](docs/aclnnScatterNdUpdate_en.md) interface. |
