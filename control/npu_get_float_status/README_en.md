# NPUGetFloatStatus

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |    ×     |
| <term>Atlas A2 products</term>    |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                               |    √     |
| <term>Atlas training products</term>                               |    √     |

## Function

* Operator function: Reads the NPU hardware floating-point overflow status register, and writes the overflow flag back to input `addr` (side effect) when overflow is detected. The output `data` is fixed to all zeros. This operator is often used together with NPUAllocFloatStatus for reading the loss scaling overflow status in mixed-precision training. This repository delivers the implementation for <term>Ascend 950PR&950DT products</term> (an empty implementation that does not yet support overflow detection): it does not modify input `addr`, and the output `data` is fixed to all zeros; implementations for other supported products are delivered by the canndev repository.

- Formula:

  $$
  \text{data} = \text{zeros}(8, \text{dtype}=\text{float32})
  $$

  When overflow is detected (status != 0), the write-back is also performed:

  $$
  \text{addr} = \text{ones}(8, \text{dtype}=\text{float32})
  $$

  Overflow detection is not yet supported on Ascend 950, and status is always 0. Therefore, `addr` is not modified, and `data` is always all zeros.

- Example:

  ```text
  Input addr:
  tensor([0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32)
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
      <td>addr</td>
      <td>Input</td>
      <td>Tensor that receives the write-back of the overflow flag, with fixed shape (8,). It is written to all ones when overflow is detected (never written back in the empty implementation on Ascend 950).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>data</td>
      <td>Output</td>
      <td>NPU floating-point status tensor, with fixed shape (8,), containing 8 float32 zero values and carrying no status information.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- This operator has one input `addr`, one output `data`, and no attributes.
- The shape of input `addr` is fixed to (8,); only the float32 dtype is supported, and only the ND format is supported.
- The shape of output `data` is fixed to (8,) and cannot be changed. The 8 elements correspond to the overflow status flag bits of the NPU hardware (32 bytes in total).
- The dtype of output `data` supports only float32, and other data types are not supported.
- The format of output `data` supports only ND.
- The value of output `data` is always all zeros (8 float32 zero values) and carries no status information; the overflow status is written back to input `addr` through the side effect.
- On Ascend 950, this is an empty implementation: overflowStatus is fixed to 0, input `addr` is not modified, the output is fixed to all zeros, there is no parallel reduction, and the result is naturally deterministic.

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
    <td><a href="./examples/test_geir_npu_get_float_status.cpp">test_geir_npu_get_float_status</a></td>
    <td>See <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
