# ScatterDiv

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

- Operator function: implements the functionality compatible with tf.compat.v1.scatter_div. The operator divides the values in the tensor updates into the slices of the tensor var by the specified index tensor indices, slice by slice. If more than one updates value applies to the same slice of var, these values are divided successively on that slice. The update is performed in place, and the output reuses the input var. The rule is as follows:

  $$
  var[indices[i], ...] = var[indices[i], ...] / updates[i, ...]
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
      <td>Empty tensors are supported. Denotes the tensor to be updated, an aclTensor on the Device side, updated in place. The shape supports 1 or more dimensions, and the data type must be consistent with updates.</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Empty tensors are supported. Denotes the index tensor to be updated, an aclTensor on the Device side. The shape supports 1 or more dimensions, and the data type is INT32 or INT64. Out-of-range indices are supported; an out-of-range index (less than 0, or greater than or equal to the first dimension of var) is silently skipped without update.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Empty tensors are supported. Denotes the tensor to be divided into var, an aclTensor on the Device side. The shape supports 1 or more dimensions and must satisfy updates.shape = indices.shape + var.shape[1:], and the data type must be consistent with var.</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>Empty tensors are supported. Denotes the updated tensor, an aclTensor on the Device side, sharing memory with the input var. Both the shape and the data type are consistent with the input var.</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>Attribute</td>
      <td>A bool value on the HOST side, indicating whether to lock the update. The default value is false.</td>
      <td>bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- Deterministic computation: the implementation is deterministic by default.
- updates value range: when the data type is an integer type (INT32, INT8, UINT8), the elements in updates cannot be 0.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface | [test_aclnn_scatter_div](examples/arch35/test_aclnn_scatter_div.cpp) | Calls the ScatterDiv operator through the [aclnnScatterDiv](docs/aclnnScatterDiv_en.md) interface. |
