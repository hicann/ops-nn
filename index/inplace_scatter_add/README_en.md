# InplaceScatterAdd

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √   |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |

## Function

- Operator function: Adds the values in updates to the first dimension of the input tensor var according to the given indices.

- Formula:

  $$
  var[indices]  +=  updates
  $$

- In-place semantics: this operator is an in-place operator. The values in updates are accumulated directly onto the storage of var, and var is overwritten in place after the call; the output var and the input var share the same storage (in-place binding), and the aclnn API internally copies the result to out for delivery.

## Parameters

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 40px">
  <col style="width: 60px">
  <col style="width: 260px">
  <col style="width: 60px">
  <col style="width: 40px">
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
      <td>Tensor to be updated, which is var in the formula. The computation result is written in place to the storage of var, and var is overwritten in place after the call (see Constraints).</td>
      <td>FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Indexes of the first dimension of var. The shape supports only 1D. The expected value range is [0, var.dim[0]); the value range is not validated: negative indices are silently skipped by the kernel-side guard, and out-of-bounds indices result in undefined behavior.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Values to be added to var. Shape constraint: dim 0 equals the number of elements in indices, and the remaining dimensions are consistent with the corresponding dimensions of var excluding dim 0.</td>
      <td>FLOAT, INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- In-place semantics: the output var and the input var share the same storage (in-place binding), and self is overwritten in place after the call.
- updates shape constraint: dim 0 of updates must equal the number of elements in indices, and the remaining dimensions must be consistent with the corresponding dimensions of var excluding dim 0; inputs that violate this constraint are rejected in the tiling phase.
- Index value range: the expected value range of indices is [0, var.dim[0]); the value range is not validated. Negative indices are silently skipped by the kernel-side guard (not accumulated), and out-of-bounds indices result in undefined behavior.
- Empty tensor support: when updates is an empty tensor, the output equals var (no accumulation is performed); when var is an empty tensor, the output is an empty tensor.
- Tensor scale: when the element scale involved in the computation does not exceed the uint32 domain (2^32-1), the 32-bit fast path is used; beyond that, the computation is automatically routed to the 64-bit variant with identical results. The actual supported tensor scale limit depends on the data type and Device memory.
- Direct kernel invocation boundary: for direct kernel invocation (bypassing the first-phase aclnn validation), boundary error codes are not committed, and the validity of the input domain is guaranteed by the caller or by the first-phase validation of the aclnn API.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------- | ---------------- | -------------------------------------------- |
| aclnn invocation  | [test_aclnn_index_add.cpp](examples/test_aclnn_index_add.cpp) | Invokes the InplaceScatterAdd operator through the [aclnnIndexAdd](docs/aclnnIndexAdd_en.md) interface. Actual dispatch behavior: the InplaceScatterAdd fast path of V1 is allowed only on the ascend910b/910_93 series through the DAV_2201 architecture gate; on Ascend 950PR&950DT (arch35), float32/int32 inputs in non-deterministic scenarios are actually executed via InplaceIndexAdd; the byte semantics of bool output are not within the supported scope of this operator (the kernel data types of this operator are FLOAT/INT32 only). |
| aclnn invocation  | [test_aclnn_index_add_v2.cpp](examples/test_aclnn_index_add_v2.cpp) | Invokes the InplaceScatterAdd operator through the [aclnnIndexAddV2](docs/aclnnIndexAddV2_en.md) interface, currently used in PTA/deterministic-computation scenarios. |
