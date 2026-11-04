# Modulate

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR&950DT products</term>|√|
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term> |    √     |

## Function

- Operator function: implements adaptive scaling (scale) and shifting (shift) of features
- Formula:

  $$
  y = x \times (1 + scaleOptional) + shiftOptional
  $$

  where $self \in R ^ {B \times L \times D}$, $scaleOptional \in R ^ {B \times 1 \times D}$, and $shiftOptional \in R ^ {B \times 1 \times D}$ (or $scaleOptional \in R ^ {B \times D}$ and $shiftOptional \in R ^ {B \times D}$)

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
      <td>Indicates the passed-in feature tensor, corresponding to x in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>Input</td>
      <td>Optional parameter, indicating the scaling coefficient, corresponding to scaleOptional in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shiftOptional</td>
      <td>Input</td>
      <td>Optional parameter, indicating the shifting coefficient, corresponding to shiftOptional in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Indicates the output tensor after scaling, corresponding to y in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_modulate](./examples/test_aclnn_modulate.cpp) | Invokes the Modulate operator through the [aclnnModulate](./docs/aclnnModulate_en.md) interface.    |
