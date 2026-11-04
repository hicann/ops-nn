# ScatterList Introduction

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
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: Applies a sparse matrix update to the variable reference.

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
      <td>Empty tensors are not supported. Denotes the tensor list to be updated. It is an aclTensor on the Device side and a dynamic input (tensor list), updated in place. Every tensor in the list must have the same shape, each tensor must have at least 1 dimension, and the data type must be consistent with updates.</td>
      <td>DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, DT_FLOAT16, DT_BF16, DT_FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indice</td>
      <td>Input</td>
      <td>Empty tensors are not supported. Denotes the write-position tensor, an aclTensor on the Device side. The shape supports 1D or 2D: when 1D, the shape is (B,) and only the write start position is given; when 2D, the shape is (B, 2) and the write start position and length are given. B is the number of tensors in the var list. The data type is INT32 or INT64.</td>
      <td>INT32, INT64.</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Empty tensors are not supported. Denotes the tensors to be updated onto var, an aclTensor on the Device side. updates must have at least 2 dimensions. The size of the first dimension equals the number of tensors in the var list, the size of the axis dimension must not exceed the corresponding axis of var, the other dimensions are consistent with var, and the data type must be consistent with var.</td>
      <td>DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, DT_FLOAT16, DT_BF16, DT_FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Input</td>
      <td>Empty tensors are not supported. Denotes the mask of the data to be updated, an aclTensor on the Device side, and is an optional input. The shape supports 1D, and the size of the first dimension equals the number of tensors in the var list. The data type is UINT8. Only values 0 and 1 are supported, and the behavior of other values is undefined.</td>
      <td>DT_UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>Empty tensors are not supported. Denotes the updated tensor list, an aclTensor on the Device side. It shares memory with the input var, and its shape and data type are consistent with the input var.</td>
      <td>DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16, DT_UINT32, DT_FLOAT16, DT_BF16, DT_FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduce</td>
      <td>Attribute</td>
      <td>A string on the HOST side, selecting the reduction operation to apply. Currently, only the value "update" is supported.</td>
      <td>string</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td>The dimension used for scatter. After normalization (a negative value is converted based on the number of dimensions of updates), the value must fall in the open interval (0, number of dimensions of updates). It must neither point to dimension 0 nor be out of bounds.</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraints

- indice semantics and value range (the values are guaranteed by the caller, and the operator performs no validation): indice gives the **write start position**, not the scatter target index. The write is a contiguous interval starting at that position, and the whole interval must not go out of the axis dimension.
  - 1D: `indice[i]` is the write start position of the i-th tensor, and the write length is the size of updates on the axis dimension.
  - 2D: `indice[i][0]` is the start position, `indice[i][1]` is the write length, and the length should not exceed the size of updates on the axis dimension.

- mask value range (guaranteed by the caller, and the operator performs no validation): only values 0 and 1 are supported. 0 means that no write is performed for that list element, and 1 means that the write is performed. The behavior of other values is undefined.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn interface | [aclnnScatterList](docs/aclnnScatterList_en.md) | Calls the ScatterList operator through the [aclnnScatterList](docs/aclnnScatterList_en.md) interface. For invocation samples, see the interface document. |
