# ForeachAddListInplace

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

- Operator function: performs element-wise x1+alpha*x2 on two tensor lists, and writes the result back to the first list in place.
- Formula:

  $$
  x1_i = x1_i + alpha \times x2_i
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
      <td>Empty tensors are supported. Denotes the first input tensor list of the addition operation, corresponding to `x1` in the formula. The data types of all tensors in this parameter are consistent, and each tensor has no more than 8 dimensions.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x1</td>
      <td>Output</td>
      <td>Empty tensors are supported. Denotes the computation result tensor list, which shares memory with input `x1`, and the computation result is written back in place. The shape and data type of each tensor in this parameter are consistent with those of the tensor at the corresponding position in input `x1`.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Empty tensors are supported. Denotes the second input tensor list of the addition operation, corresponding to `x2` in the formula. The data type, data format, and shape are consistent with the data type, data format, and shape of input parameter `x1`. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Empty tensors are not supported. Denotes the coefficient of the second input in the addition operation, corresponding to `alpha` in the formula. The number of elements is 1. The data type has a defined relationship with the data type of input parameter `x1`: when the data type of `x1` is FLOAT32, FLOAT16, or INT32, the data type is consistent with the data type of `x1`; when the data type of `x1` is BFLOAT16, the data type supports FLOAT32.</td>
      <td>FLOAT32, FLOAT16, INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The numbers of tensors in input parameters `x1` and `x2` must be the same, and the number of tensors in a single tensor list does not exceed 256.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn invocation | [test_aclnn_foreach_add_list_inplace](examples/arch35/test_aclnn_foreach_add_list_inplace.cpp) | Invokes the ForeachAddListInplace operator through the [aclnnForeachAddListInplace](docs/aclnnForeachAddListInplace_en.md) interface. |
