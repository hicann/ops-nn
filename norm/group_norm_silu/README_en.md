# GroupNormSiluV2

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    ×     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: Computes the group normalization result out of the input self, the mean meanOut, the reciprocal of the standard deviation rstdOut, and the output of silu.

- Formula:
  - **GroupNorm:**

    Let $E[x] = \bar{x}$ denote the mean of $x$, and $Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$ denote the variance of $x$. Then

    $$
    \left\{
    \begin{array} {rcl}
    out& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
    meanOut& &= E[x]\\
    rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
    \end{array}
    \right.
    $$

  - **Silu:**

    $$
    out = \frac{x}{1+e^{-x}}
    $$

    When activateSilu is True, Silu is computed. In this case, the x in the Silu formula is the out obtained by the GroupNorm formula.

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
      <td>self</td>
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
      <td>group</td>
      <td>Attribute</td>
      <td>The second dimension of self is divided into group groups.</td>
      <td>INT32, INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td>eps value in the computation formulas of out and rstdOut.</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateSilu</td>
      <td>Attribute</td>
      <td>Specifies whether the silu computation is supported.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor out.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>meanOut</td>
      <td>Output</td>
      <td>Output tensor mean.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstdOut</td>
      <td>Output</td>
      <td>Output tensor rstd.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: The data types of all inputs and outputs do not support BFLOAT16.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_group_norm_silu.cpp](examples/test_aclnn_group_norm_silu.cpp) | Invoke the GroupNormSilu operator through the [aclnnGroupNormSilu.md](docs/aclnnGroupNormSilu_en.md) API. |
| aclnn API  | [test_aclnn_group_norm_silu_v2.cpp](examples/test_aclnn_group_norm_silu_v2.cpp) | Invoke the GroupNormSilu operator through the [aclnnGroupNormSiluV2.md](docs/aclnnGroupNormSiluV2_en.md) API. |
| Graph mode | -  | Invoke the GroupNormSilu operator by building a graph through the [operator IR](op_graph/group_norm_silu_proto.h). |
