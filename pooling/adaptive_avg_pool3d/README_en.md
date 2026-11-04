# AdaptiveAvgPool3d

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |
| Atlas 200I/500 A2 inference products | × |
| Atlas inference products | √ |
| Atlas training products | × |
| Kirin X90 processor series products | √ |
| Kirin 9030 processor series products | √ |

## Function

- Operator function: When the specified 3D output shape information is given, perform 3D adaptive average pooling on the input tensor.

## Parameters

<table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 200px">
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
      <td>Input parameter for the adaptiveavgpool3d computation. It indicates the target tensor to be converted.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attribute</td>
      <td>Input parameter for the adaptiveavgpool3d computation. It indicates the shape size of the output in the DHW dimensions.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter of the adaptiveavgpool3d computation. It indicates the converted result tensor.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas inference products: BFLOAT16 is not supported.
- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraints

None.

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_adaptive_avg_pool3d.cpp](examples/test_aclnn_adaptive_avg_pool3d.cpp) | Invoke the AdaptiveAvgPool3d operator through the [aclnnAdaptiveAvgPool3d](docs/aclnnAdaptiveAvgPool3d_en.md) API. |
