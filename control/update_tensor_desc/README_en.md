# UpdateTensorDesc

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

* Operator function: updates the description information (tensor_desc) of the output tensor. The operator receives a placeholder input `x` (the kernel does not read its data) and performs an RMW (read-modify-write) on output `y` according to the mandatory attribute `shape`: it writes the target number of dimensions and the size of each dimension into fixed slots of the 128×int64 description buffer corresponding to `y`, and the other slots keep their original values. It is commonly used to update the output TensorDesc in dynamic shape scenarios.

- Formula:

  Let $N = len(shape)$, and output `y` be a 128×int64 description buffer:

  $$
  y[3] = N,\quad y[4+i] = shape[i],\ i = 0,\dots,N-1
  $$

  The other slots of `y` (subscripts 0~2 and 4+N~127) keep their original values unchanged.

- Example:

  ```text
  Input x (placeholder input, its data does not participate in the computation):
  tensor([1., 1., 1., 1., 1., 1., 1., 1.], dtype=torch.float32)
  Attribute shape:
  [4, 8, 4]
  Output y (after RMW, only the first 8 elements are shown):
  tensor([1, 1, 1, 3, 4, 8, 4, 1], dtype=torch.int64)
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
      <td>x</td>
      <td>Input</td>
      <td>Placeholder input tensor. Its data does not participate in the computation, and the kernel does not read its data.</td>
      <td>bool/float16/float/float64/int8/int16/int32/int64/uint8/uint16/uint32/uint64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Description buffer tensor. Its dtype is always int64, and its shape is derived from the attribute `shape`.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shape</td>
      <td>Attribute</td>
      <td>Target shape, mandatory, of the ListInt type.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- This operator has one input `x`, one output `y`, and one mandatory attribute `shape`.
- Input `x`: rank ∈ [0, 8], its dtype supports 12 types in total: bool/float16/float/float64/int8/int16/int32/int64/uint8/uint16/uint32/uint64, and its format supports only ND.
- Output `y`: its dtype is always int64, its format supports only ND, and its shape is derived from the attribute `shape`.
- Attribute `shape`: rank ∈ [1, 124], each element is a non-negative integer, and the product of all elements (numel) ≥ 128.
- The total number of elements (numel) of output `y` ≥ 128: the kernel always performs RMW on the whole 128×int64 block. When numel is greater than 128, only the first 128 elements are read and written, and the observable effect is equivalent to covering only the subscript range [3, 3+N).
- The non-written slots of output `y` (subscripts 0~2 and 4+N~127) keep and pass through their original values before execution.

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
    <td><a href="./examples/test_geir_update_tensor_desc.cpp">test_geir_update_tensor_desc</a></td>
    <td>See <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
