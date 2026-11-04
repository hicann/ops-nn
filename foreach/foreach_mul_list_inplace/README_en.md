# ForeachMulListInplace

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

- Operator function: Performs element-wise multiplication on two tensor lists and writes the result in place back to the first list.
- Formula:

  $$
  x1_i = x1_i \times x2_i
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
      <td>x1</td>
      <td>Input</td>
      <td>Empty tensors are supported. Indicates the first input tensor list of the multiplication, corresponding to `x1` in the formula. The data types of all tensors in this parameter are consistent, and each tensor has no more than 8 dimensions.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x1</td>
      <td>Output</td>
      <td>Empty tensors are supported. Indicates the computation result tensor list, which shares memory with the input `x1`, and the computation result is written back in place. The shape and data type of each tensor in this parameter are consistent with the tensor at the corresponding position in the input `x1`.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Empty tensors are supported. Indicates the second input tensor list of the multiplication, corresponding to `x2` in the formula. Its data type, data format, and shape are consistent with the data type, data format, and shape of the input parameter `x1`. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The number of tensors in the input parameters `x1` and `x2` must be the same, and the number of tensors contained in a single tensor list must not exceed 256.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn invocation | [test_aclnn_foreach_mul_list_inplace](examples/arch35/test_aclnn_foreach_mul_list_inplace.cpp) | Calls the ForeachMulListInplace operator through the [aclnnForeachMulListInplace](docs/aclnnForeachMulListInplace_en.md) interface. |
