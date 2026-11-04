# ThresholdV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: Perform the threshold operation on input x. When an element in x is greater than threshold, the element is returned. Otherwise, value is returned.

- Formula:

$$
out(x) = \begin{cases} value, & x \le threshold \\ x, & otherwise \end{cases}
$$

## Parameters

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>Corresponding to x in the formula</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, INT64, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>Input</td>
      <td>Corresponding to threshold in the formula</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, INT64, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>Corresponding to value in the formula</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, INT64, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor in the formula</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT8, INT32, INT64, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Sample code | Description |
| -----------| -----------------------------------------------------------------------------| ----------------------------------------------------------------------------------------------------------|
| Graph mode | [test_geir_threshold_v2.cpp](./examples/test_geir_threshold_v2.cpp) | Invoke the ThresholdV2 operator through graph construction using the [operator IR](./op_graph/threshold_v2_proto.h). |
| aclnn invocation | [test_aclnn_threshold](./examples/test_aclnn_threshold.cpp) | Calls the operator through the [aclnnThreshold&aclnnInplaceThreshold](./docs/aclnnThreshold&aclnnInplaceThreshold_en.md) API. |
| aclnn invocation | [test_aclnn_inplace_threshold](./examples/test_aclnn_inplace_threshold.cpp) | Calls the operator through the [aclnnThreshold&aclnnInplaceThreshold](./docs/aclnnThreshold&aclnnInplaceThreshold_en.md) API (the result is written back to the input tensor). |
