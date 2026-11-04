# Quantize

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
| <term>Atlas inference products</term>    |     √    |
| <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Performs quantization processing on the input tensor x.

- Formula:

  $$
  out=round((x/scales)+zeroPoints)
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
      <td>x in the formula</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Input</td>
      <td>scales in the formula</td>
      <td>FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points</td>
      <td>Input</td>
      <td>zeroPoints in the formula</td>
      <td>FLOAT, INT32, INT8, UINT8, BFLOAT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor out in the formula</td>
      <td>INT8, UINT8, INT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_quantize.cpp](examples/test_aclnn_quantize.cpp) | Invoke the operator through the [aclnnQuantize.md](docs/aclnnQuantize_en.md) API. |
