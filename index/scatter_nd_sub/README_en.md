# ScatterNdSub

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function

- Operator function: ScatterNdSub subtracts the values in updates from var element by element at the index positions specified by indices, and writes the result back to var (in-place).

- Formula:

$$
\text{output}[\text{indices}[i]] = \text{var}[\text{indices}[i]] - \text{updates}[i]
$$

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
      <td>var</td>
      <td>Input</td>
      <td>Parameter tensor to be operated on.</td>
      <td>FLOAT16, FLOAT, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Index tensor, which specifies the positions where updates need to be subtracted.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Values to be subtracted. The shape matches the prefix dimensions of indices.</td>
      <td>FLOAT16, FLOAT, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>Updated parameter tensor, with the same shape and dtype as the input var.</td>
      <td>FLOAT16, FLOAT, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>Attribute</td>
      <td>Whether to use a lock to protect variable updates (the lock mechanism is not implemented on the NPU). Default value: false.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The rank of indices must be greater than or equal to 1, that is, indices is a tensor of the form (N, ..., K) (N is the number of index groups, and the last dimension K is the index depth). 1D indices are supported (in this case, N=1, that is, a single index with depth K).
- The rank of var must be greater than or equal to the last dimension of indices.
- The shape of updates must be consistent with indices.shape[:-1] + var.shape[indices.shape[-1]:].
- Out-of-bounds index values are not supported.

## Invocation Description

| Invocation method | Invocation sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_scatter_nd_sub](./examples/test_geir_scatter_nd_sub.cpp) | Calls the scatter_nd_sub operator on the host side through graph construction with GE IR. |
