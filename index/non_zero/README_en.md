# NonZero

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √  |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×    |
| <term>Atlas training products</term>                              |    ×   |

## Function

Operator function: Finds the positions of non-zero elements in `self`. If the number of dimensions of self is D and the number of non-zero elements in self is N, the shape of the returned `out` is N * D, where each column represents the position coordinates of a non-zero element.

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
      <td>Represents the target tensor to be computed, an aclTensor on the Device side.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, BFLOAT16, UINT16, UINT32, UINT64.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The computation output, an aclTensor on the Device side.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Attribute</td>
      <td>Specifies whether the output needs to be transposed.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas training products</term>, <term>Atlas inference products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Due to hardware resource limits, the output indices are required to be within the int32 precision range, so a dimension of the input cannot exceed the representation range of int32.

## Constraints

- Deterministic Computation:
  - aclnnNonzero is deterministic by default.
- The output size of the interface needs to be requested according to the maximum output size (that is, the scenario where all elements are non-zero; tests show that the output size cannot exceed 2G). Meanwhile, the workspace used by this interface is also large, so you need to pay attention to whether it exceeds the device memory size.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_non_zero](./examples/test_aclnn_non_zero.cpp) | Calls the Nonzero operator through the [aclnnNonzero](./docs/aclnnNonzero_en.md) interface. |
| aclnn invocation | [test_aclnn_non_zero_v2](./examples/test_aclnn_non_zero_v2.cpp) | Calls the NonzeroV2 operator through the [aclnnNonzeroV2](./docs/aclnnNonzeroV2_en.md) interface. |
