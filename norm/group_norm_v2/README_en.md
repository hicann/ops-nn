# GroupNormV2

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR&950DT products</term>                   |     √     |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>    |     √   |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Description: Computes the group normalization result out, the mean meanOut, and the reciprocal of the standard deviation rstdOut of the input x.

- Formula:
  - **GroupNorm:**
  Let $E[x] = \bar{x}$ denote the mean of $x$, and let $Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$ denote the variance of $x$. Then:

  $$
  \left\{
  \begin{array} {rcl}
  y& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
  mean& &= E[x]\\
  rstd& &= \frac{1}{\sqrt{Var[x] + eps}}\\
  \end{array}
  \right.
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
      <td>x</td>
      <td>Input</td>
      <td>x in the computation formula of out.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>gamma in the computation formula of out.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>beta in the computation formula of out.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_groups</td>
      <td>Attribute</td>
      <td>Dimension 1 of x is divided into group groups.</td>
      <td>INT32, INT64</td>
      <td>-</td>
    </tr>
     <tr>
      <td>data_format</td>
      <td>Optional attribute</td>
      <td>Input data format. "NHWC" is supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Optional attribute</td>
      <td>The eps value in the computation formulas of out and rstdOut. Default value: 0.00001f.</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_training</td>
      <td>Optional attribute</td>
      <td>Marks whether the scenario is the training scenario. true indicates the training scenario, and false indicates the inference scenario. Default value: true.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Output</td>
      <td>mean output tensor.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Output</td>
      <td>rstd output tensor.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_group_norm_v2](./examples/test_geir_group_norm_v2.cpp)   | Invokes the GroupNormV2 operator by building a graph through the [operator IR](./op_graph/group_norm_v2_proto.h). |
