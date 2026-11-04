# LogitGrad

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
the backward propagation of [aclnnLogit](../logit/docs/aclnnLogit_en.md).
- Computation formula:

$$
dx_i=
\begin{cases}
NaN, & \text{if } x < \text0 \text{ or } x > 1 ,eps <0 \\
0, & \text{if } x < \text{eps} \text{ or } x > 1 - \text{eps},eps \geq 0 \\
\frac{dy_i}{x_i \cdot (1 - x_i)}, & \text{if } \text{eps} \leq x_i \leq 1 - \text{eps} \\
\end{cases}
$$

## Parameters

<table style="undefined;table-layout: fixed; width: 908px"><colgroup>
  <col style="width: 91px">
  <col style="width: 149px">
  <col style="width: 296px">
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
      <td>x</td>
      <td>Input</td>
      <td>x in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>dy in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td><ul><li>eps limit boundary of input x, which prevents division-by-0 errors.</li><li>The recommended value is -1.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>dx in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Impact of the value of eps on x and the output:
  - When eps is smaller than 0 and the value range of x is not within [0, 1], the output is NaN.
  - When eps is greater than or equal to 0 and the value range of x is not within [eps, 1 - eps], the output is 0.
  - When eps is greater than 1, the output is NaN; when the value is 1, the output is Inf.
  - When eps is Inf, the output is 0.
  - When eps is NaN, the output is NaN.
- Impact of the value of x on the output:
  - When x is 0 or 1, the output is Inf.
  - When x is Inf or NaN, the output is NaN.
- When dy is Inf or NaN, the output is Inf or NaN.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_logit_grad](./examples/test_aclnn_logit_grad.cpp) | Calls the LogitGrad operator through the [aclnnLogitGrad](./docs/aclnnLogitGrad_en.md) interface.    |
| Graph-mode invocation | -   | Calls the LogitGrad operator by constructing a graph through the [operator IR](./op_graph/logit_grad_proto.h). |
