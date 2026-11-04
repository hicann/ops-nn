# AdaptiveAvgPool2d

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|√|
|Atlas training products|×|
|Kirin X90 processor series products|√|
|Kirin 9030 processor series products|√|

## Function

- Operator function: Performs the 2D adaptive average pooling computation on the input tensor when the specified 2D output shape information is given.

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
      <td>Input parameter for the adaptiveavgpool2d computation. Represents the target tensor to be converted.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attribute</td>
      <td>Input parameter for the adaptiveavgpool2d computation. Represents the shape size of the output on the HW dimensions.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter of the adaptiveavgpool2d computation. Represents the result tensor after conversion.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas inference products: BFLOAT16 is not supported.
- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_adaptive_avg_pool2d.cpp](examples/arch35/test_aclnn_adaptive_avg_pool2d.cpp) | Calls the AdaptiveAvgPool2d operator through the [aclnnAdaptiveAvgPool2d](docs/aclnnAdaptiveAvgPool2d_en.md) API. |
