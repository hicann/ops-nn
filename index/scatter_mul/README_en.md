# ScatterMul

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: it implements a function compatible with tf.compat.v1.scatter_mul, multiplying the values in the tensor updates slice by slice onto the slices of the tensor var according to the specified index tensor indices. If more than one updates value acts on the same slice of var, these values are multiplied onto that slice successively. It is an in-place update, and the output reuses the input var. The rule is as follows:

  $$
  var[indices[i], ...] = var[indices[i], ...] \times updates[i, ...]
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
      <td>Empty tensors are supported. It denotes the tensor to be updated, an aclTensor on the Device side, updated in place. The shape supports 1 dimension or more, and its data type must be the same as updates. The strides must be contiguous: a contiguous view with a storage offset is supported, while broadcast views and strided views are not.</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Empty tensors are supported. It denotes the index tensor to be updated, an aclTensor on the Device side. The shape supports 0 dimension or more, including a scalar index (a 0-dimensional tensor, or a tensor of shape (1,) when the updates shape equals var.shape[1:]), and the data type is INT32 or INT64. Out-of-bounds indexing is supported. Out-of-bounds indices (less than 0 or greater than or equal to the first dimension of var) are silently skipped and do not trigger an update.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Empty tensors are supported. It denotes the tensor to be multiplied onto var, an aclTensor on the Device side. The shape supports 1 dimension or more and must satisfy updates.shape = indices.shape + var.shape[1:], and its data type must be the same as var.</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>Empty tensors are supported. It denotes the updated tensor, an aclTensor on the Device side, sharing memory with the input var. Its shape and data type are both the same as the input var.</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>Attribute</td>
      <td>A bool value on the HOST side, whether to lock the update. Default value: false.</td>
      <td>bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- Deterministic computation: the deterministic implementation is used by default.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn API | [test_aclnn_scatter_mul](examples/arch35/test_aclnn_scatter_mul.cpp) | Calls the ScatterMul operator through the [aclnnScatterMul](docs/aclnnScatterMul_en.md) API. |
