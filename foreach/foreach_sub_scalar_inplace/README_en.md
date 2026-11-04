# ForeachSubScalarInplace

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | × |
|  <term>Kirin 9030 processor series products</term> | × |

## Function

- Description: Subtracts the same scalar element-wise from each tensor in the input tensor list (in place).
- Formula:

  $$
  x_i = x_i - scalar
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Empty tensors are supported. Denotes the input tensor list for the subtraction computation, corresponding to `x` in the formula. The data types of all tensors in this parameter are consistent, and the number of dimensions (Dim) of each tensor is no greater than 8.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td>Empty tensors are supported. Denotes the computation result tensor list, which shares memory with the input `x`, and the computation result is written back in place. The shape and data type of each tensor in this parameter are consistent with those of the tensor at the corresponding position in the input `x`.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalar</td>
      <td>Input</td>
      <td>Empty tensors are not supported. Denotes the input tensor for the subtraction computation, corresponding to `scalar` in the formula. The number of elements is 1, and the number of dimensions (Dim) is no greater than 8. The data type has a certain correspondence with the data type of the input parameter `x`: when the data type of `x` is FLOAT32 or INT32, the data type is consistent with that of `x`; when the data type of `x` is BFLOAT16, the data type supports FLOAT32; when the data type of `x` is FLOAT16, the data type supports FLOAT16 and FLOAT32.</td>
      <td>FLOAT32, FLOAT16, INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- A single tensor list of the input parameter `x` contains no more than 256 tensors.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn interface | [test_aclnn_foreach_sub_scalar_inplace](examples/arch35/test_aclnn_foreach_sub_scalar_inplace.cpp) | Invokes the ForeachSubScalarInplace operator through the [aclnnForeachSubScalarInplace](docs/aclnnForeachSubScalarInplace_en.md) interface. |
