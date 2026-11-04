# ForeachDivListInplace

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

- Operator function: Performs element-wise division on two tensor lists, and writes the result back in place to the first list.
- Formula:

  $$
  x1_i = x1_i / x2_i
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
      <td>Empty tensors are supported. Represents the first input tensor list of the division operation, which corresponds to `x1` in the formula. All tensors in this parameter have the same data type, and the number of dimensions of each tensor does not exceed 8.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x1</td>
      <td>Output</td>
      <td>Empty tensors are supported. Represents the tensor list of the computation result, which shares memory with the input `x1`, and the computation result is written back in place. The shape and data type of each tensor in this parameter are the same as those of the tensor at the corresponding position in the input `x1`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Empty tensors are supported. Represents the second input tensor list (divisor) of the division operation, which corresponds to `x2` in the formula. The data type, data format, and shape are the same as the data type, data format, and shape of the input parameter `x1`. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The input parameters `x1` and `x2` must contain the same number of tensors, and a single tensor list contains no more than 256 tensors.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn API | [test_aclnn_foreach_div_list_inplace](examples/arch35/test_aclnn_foreach_div_list_inplace.cpp) | Calls the ForeachDivListInplace operator through the [aclnnForeachDivListInplace](docs/aclnnForeachDivListInplace_en.md) API. |
