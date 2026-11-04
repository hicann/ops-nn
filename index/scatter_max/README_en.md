# ScatterMax

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

- Operator function: Implements a function compatible with tf.compat.v1.scatter_max. It updates the values in the tensor updates onto the slices of the tensor var by taking the elementwise maximum value according to the specified index tensor indices. If more than one updates value acts on the same slice of var, the maximum value is taken on that slice in turn. It is an in-place update, and the output reuses the input var. The rule is as follows:

  $$
  var[indices[i], ...] = \max(var[indices[i], ...], updates[i, ...])
  $$

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
      <td>Empty tensors are supported. Denotes the tensor to be updated, an aclTensor on the Device side, updated in place. The shape supports 1D and above, and the data type must be consistent with updates.</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Empty tensors are supported. Denotes the index tensor to be updated, an aclTensor on the Device side. The shape supports 1D and above, and the data type is INT32 or INT64. Out-of-bounds indices are supported. Out-of-bounds indices (less than 0 or greater than or equal to the first dimension of var) are silently skipped without update.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Empty tensors are supported. Denotes the tensor that takes the elementwise maximum value with var, an aclTensor on the Device side. The shape supports 1D and above, and must satisfy updates.shape = indices.shape + var.shape[1:]. The data type must be consistent with var.</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>Empty tensors are supported. Denotes the updated tensor, an aclTensor on the Device side, sharing memory with the input var. Its shape and data type are consistent with the input var.</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>Attribute</td>
      <td>A bool value on the HOST side, specifying whether to lock the update. The default value is false.</td>
      <td>bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

- <term>Ascend 950PR&950DT products</term>: The data types of var and updates support FLOAT16, FLOAT32, INT32, INT8, and UINT8.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data types of var and updates support FLOAT16, FLOAT32, INT32, INT64, and DOUBLE, and do not support INT8 or UINT8.

## Constraints

- Deterministic computation: The default implementation is deterministic.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface | [test_aclnn_scatter_max](examples/arch35/test_aclnn_scatter_max.cpp) | Calls the ScatterMax operator through the [aclnnScatterMax](docs/aclnnScatterMax_en.md) interface. |
