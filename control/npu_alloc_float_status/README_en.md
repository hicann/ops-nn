# NPUAllocFloatStatus

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                               |    √     |
| <term>Atlas training products</term>                               |    √     |

## Function

* Operator function: Allocates a tensor of 8 float32 zero values, which is used for NPU overflow status detection. This operator has no input, and the output is an all-zero float32 tensor with the fixed shape (8,). It is commonly used in the loss scaling overflow status allocation scenario in mixed-precision training.

- Formula:

  $$
  \text{output} = \text{zeros}(8, \text{dtype}=\text{float32})
  $$

- Example:

  ```text
  Output data:
  tensor([0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32)
  ```

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
      <td>data</td>
      <td>Output</td>
      <td>NPU float status tensor, with the fixed shape (8,) and containing 8 float32 zero values, used for overflow status detection.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- This operator has no inputs and no attributes, and has only one output `data`.
- The shape of the output `data` is fixed to (8,) and cannot be changed. The 8 elements correspond to the overflow status flags of the NPU hardware (32 bytes in total).
- The dtype of the output `data` supports only float32. Other data types are not supported.
- The format of the output `data` supports only ND.
- The output values are always all zeros (8 float32 zero values) and do not depend on any input.
- This operator is a deterministic operator. The output is fixed all zeros, with no parallel reduction, so it is deterministic by nature.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/test_geir_npu_alloc_float_status.cpp">test_geir_npu_alloc_float_status</a></td>
    <td>See <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
