# ForeachAddcmulScalar

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: First perform element-wise multiplication on tensor list x2 and tensor list x3, then multiply the result by tensor scalar, and finally perform element-wise addition between the computed result and tensor list x1.
- Formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}], x3 = [{x3_0}, {x3_1}, ... {x3_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = x1_{i} + {\rm scalar} \times x2_{i} \times x3_{i} (i=0,1,...n-1)
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
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>First input tensor list of the addition in the mixed operation, corresponding to `x1` in the formula. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Second input tensor list of the multiplication in the mixed operation, corresponding to `x2` in the formula. The data type, data format, and shape are the same as those of input parameter `x1`, and all tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>Input</td>
      <td>Third input tensor list of the multiplication in the mixed operation, corresponding to `x3` in the formula. The data type, data format, and shape are the same as those of input parameter `x1`, and all tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalar</td>
      <td>Input</td>
      <td>First input tensor of the multiplication in the mixed operation, corresponding to `scalar` in the formula. The number of elements is 1.</td>
      <td>FLOAT32, FLOAT16, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor list of the mixed operation, corresponding to `y` in the formula. The data type and data format are the same as those of input parameter `x1`, and the shape size is greater than or equal to the shape size of input parameter `x1`. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - When the data type of `x1` is FLOAT32, FLOAT16, or INT32, the data type of `scalar` is the same as that of `x1`.
  - When the data type of `x1` is BFLOAT16, the supported data type of `scalar` is FLOAT32.
- <term>Ascend 950PR&950DT products</term>:
  - When the data type of `x1` is FLOAT32 or INT32, the data type of `scalar` is the same as that of `x1`.
  - When the data type of `x1` is BFLOAT16, the supported data type of `scalar` is FLOAT32.
  - When the data type of `x1` is FLOAT16, the supported data types of `scalar` are FLOAT16 and FLOAT32.

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- Non-contiguous tensors are not supported for the output.

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_addcmul_scalar](examples/test_aclnn_foreach_addcmul_scalar.cpp) | Invoke the ForeachAddcmulScalar operator through the [aclnnForeachAddcmulScalar](docs/aclnnForeachAddcmulScalar_en.md) API. |
| aclnn API  | [test_aclnn_foreach_addcmul_scalar_v2](examples/test_aclnn_foreach_addcmul_scalar_v2.cpp) | Invoke the ForeachAddcmulScalar operator through the [aclnnForeachAddcmulScalarV2](docs/aclnnForeachAddcmulScalarV2_en.md) API. |
| Graph mode | -  | Invoke the ForeachAddcmulScalar operator through graph construction using the [operator IR](op_graph/foreach_addcmul_scalar_proto.h).         |
