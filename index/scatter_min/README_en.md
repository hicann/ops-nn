# ScatterMin

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

- Operator function: implements the function compatible with tf.compat.v1.scatter_min. The operator updates the values in the tensor updates element-wise to the slices of the tensor var with the minimum value, according to the specified index tensor indices. If more than one updates value acts on the same slice of var, the minimum value is taken on that slice successively. It is an in-place update, and the output reuses the input var. The rule is as follows:

  $$
  var[indices[i], ...] = \min(var[indices[i], ...], updates[i, ...])
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
      <td>Supports empty tensors. Represents the tensor to be updated, an aclTensor on the Device side, updated in place. Shapes with 1 or more dimensions are supported, and the data type must be consistent with updates.</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Supports empty tensors. Represents the index tensor for the update, an aclTensor on the Device side. Shapes with 1 or more dimensions are supported, and the data type is INT32 or INT64. Out-of-bounds indices are supported. Out-of-bounds indices (less than 0 or greater than or equal to the first dimension of var) are silently skipped without any update.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Supports empty tensors. Represents the tensor for the element-wise minimum with var, an aclTensor on the Device side. Shapes with 1 or more dimensions are supported, and updates.shape = indices.shape + var.shape[1:] must be satisfied. The data type must be consistent with var.</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>Supports empty tensors. Represents the updated tensor, an aclTensor on the Device side, which shares memory with the input var. The shape and data type are both consistent with the input var.</td>
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

- <term>Ascend 950PR&950DT products</term>: the data types of var and updates support FLOAT16, FLOAT32, INT32, INT8, and UINT8.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: the data types of var and updates support FLOAT16, FLOAT32, INT32, INT64, and DOUBLE, and INT8 and UINT8 are not supported.

## Constraints

- Deterministic Computation: deterministic by default.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation | [test_aclnn_scatter_min](examples/arch35/test_aclnn_scatter_min.cpp) | Invoke the ScatterMin operator through the [aclnnScatterMin](docs/aclnnScatterMin_en.md) interface. |
