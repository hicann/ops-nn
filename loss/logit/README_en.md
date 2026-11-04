# Logit

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function:

  This operator is a mathematical operation that converts a probability to log-odds, and it is commonly used for the inverse transformation of probability values. It converts a probability value between 0 and 1 into a real number.

- Formula:

  $$
  y_i = \ln \left( \frac{z_i}{ 1 -z_i} \right)
  $$

  Here:

  $$
  z_i =
  \begin{cases}
  \text{eps}, & \text{if } x_i < \text{eps} \\
  x_i, & \text{if } \text{eps} \leq x_i \leq 1 - \text{eps} \\
  1 - \text{eps}, & \text{if } x_i > 1 - \text{eps}
  \end{cases}
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 878px"><colgroup>
  <col style="width: 91px">
  <col style="width: 149px">
  <col style="width: 266px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>input</td>
      <td>Input</td>
      <td>Input input in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td><ul><li>eps limit boundary of the input x, preventing division-by-zero errors.</li><li>Suggested value: -1.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>y in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- When the value range of input is not within [0, 1], the output is NaN; the output is -Inf for the value 0 and Inf for the value 1.
- When eps is greater than 1, the output is NaN, and the output is Inf for the value 1.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_logit](./examples/test_aclnn_logit.cpp) | Invokes the Logit operator through the [aclnnLogit](./docs/aclnnLogit_en.md) interface.    |
| Graph-mode invocation | -   | Invokes the Logit operator by building a graph through the [operator IR](./op_graph/logit_proto.h). |
