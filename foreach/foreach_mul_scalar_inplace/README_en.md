# ForeachMulScalarInplace

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

- Operator function: Multiplies each tensor in the input tensor list element-wise by the same scalar (in-place).
- Formula:

  $$
  x_i = x_i \times scalar
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
      <td>Empty tensors are supported. Represents the input tensor list of the multiplication, corresponding to `x` in the formula. All tensors in this parameter have the same data type, and the number of dimensions (Dim) of each tensor is not greater than 8.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td>Empty tensors are supported. Represents the computation result tensor list. It shares memory with input `x`, and the computation result is written back in place. The shape and data type of each tensor in this parameter are the same as those of the tensor at the corresponding position in input `x`.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalar</td>
      <td>Input</td>
      <td>Empty tensors are not supported. Represents the input tensor of the multiplication, corresponding to `scalar` in the formula. It has 1 element and its number of dimensions (Dim) is not greater than 8. Its data type has a certain correspondence with the data type of input parameter `x`: when the data type of `x` is FLOAT32 or INT32, its data type is the same as the data type of `x`; when the data type of `x` is BFLOAT16, the data type supports FLOAT32; when the data type of `x` is FLOAT16, the data types support FLOAT16 and FLOAT32.</td>
      <td>FLOAT32, FLOAT16, INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The number of tensors contained in a single tensor list of input parameter `x` does not exceed 256.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn interface | [test_aclnn_foreach_mul_scalar_inplace](examples/arch35/test_aclnn_foreach_mul_scalar_inplace.cpp) | Invokes the ForeachMulScalarInplace operator through the [aclnnForeachMulScalarInplace](docs/aclnnForeachMulScalarInplace_en.md) interface. |
