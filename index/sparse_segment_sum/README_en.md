# SparseSegmentSum

English | [简体中文](./README.md)

## Applicable Products

| Product                                                     | Support |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                  |    √     |
| <term>Atlas A3 products</term>|    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                 |    ×     |
| <term>Atlas inference products</term>                         |    √     |
| <term>Atlas training products</term>                         |    √     |

## Function

- Description: Sums the slices of `x` along the sparse segments specified by `segment_ids`.
- Formula: `y[segment_ids[i]] += x[indices[i]]`.

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
      <td>x</td>
      <td>Input</td>
      <td>Input tensor, with a rank of at least 1.</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, DOUBLE, FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>1D index tensor, used to select slices on dimension 0 of `x`.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>segment_ids</td>
      <td>Input</td>
      <td>1D segment ID tensor, whose shape must be consistent with that of `indices`, with values that must be non-negative and monotonically non-decreasing.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, with the data type the same as that of `x`.</td>
      <td>INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, DOUBLE, FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The rank of `x` must be greater than or equal to 1.
- `indices` and `segment_ids` must be 1D tensors with the same number of elements.
- `segment_ids` must be arranged in non-decreasing order.
- The elements in `indices` must be smaller than the size of dimension 0 of `x`.

## Invocation Description

| Invocation method | Sample code | Description |
| -------- | -------- | ---- |
| Graph mode | [test_geir_sparse_segment_sum](examples/test_geir_sparse_segment_sum.cpp) | Invokes the SparseSegmentSum operator by building a graph through the GE IR. |
