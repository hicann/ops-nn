# ScatterNdMin

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: According to indices, compute the minimum value between updates and a single value or slice within a given variable.
- Formula: ref[indices[k]]=min(ref[indices[k]], updates[k])
    - ref is a P-dimensional tensor;
    - indices is a Q-dimensional integer tensor;
    - The shape of indices must be [d0,..., d_Q-2, K], where 0<K<=P;
    - updates is a tensor with Q−1+P−K dimensions: [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]]

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
      <td>Tensor to be updated, equivalent to `ref` in the formula.</td>
      <td>DT_INT8, DT_INT16, DT_INT64, DT_UINT64, DT_BOOL, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT32, DT_UINT32, DT_UINT8, DT_UINT16, DT_DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Index tensor that indexes into `ref` in the formula.</td>
      <td>INT32, INT64.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Tensor used to update the var tensor. It must have the same data type as input tensor var.</td>
      <td>DT_INT8, DT_INT16, DT_INT64, DT_UINT64, DT_BOOL, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT32, DT_UINT32, DT_UINT8, DT_UINT16, DT_DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>Updated tensor.</td>
      <td>DT_INT8, DT_INT16, DT_INT64, DT_UINT64, DT_BOOL, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT32, DT_UINT32, DT_UINT8, DT_UINT16, DT_DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>Optional attribute</td>
      <td>Optional attribute. Default value: `false`. If the value is `true`, this operation is protected by a lock.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- Input shape restrictions:
  - indices must be at least 1D, and the size of its last dimension cannot exceed the dimension size of varRef.
  - Assume that the size of the last dimension of indices is a. The shape of updates is then equal to the shape of indices excluding the last dimension plus the shape of varRef excluding the first a dimensions. Example: if the shape of varRef is (4, 5, 6) and the shape of indices is (3, 2), the shape of updates must be (3, 6).

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_scatter_nd_min](examples/test_geir_scatter_nd_min.cpp) | Invoke the ScatterNdMin operator through GE IR graph construction. |
