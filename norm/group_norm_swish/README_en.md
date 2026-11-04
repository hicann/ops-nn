# GroupNormSwish

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |
| <term>Kirin X90 processor series products</term>                        | √ |
| <term>Kirin 9030 processor series products</term>                       | √ |

## Function

- Operator function: computes the group normalization result out of the input x, the mean meanOut, the reciprocal of the standard deviation rstdOut, and the output of Swish.

- Formula:
  - **GroupNorm:**

    Let $E[x] = \bar{x}$ denote the mean of $x$ and $Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$ denote the variance of $x$. Then

    $$
    \left\{
    \begin{array} {rcl}
    yOut& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * gamma + \beta \\
    meanOut& &= E[x]\\
    rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
    \end{array}
    \right.
    $$

  - **Swish:**

    $$
    yOut = \frac{x}{1+e^{-scale * x}}
    $$

    When activateSwish is True, Swish is computed, and the x in the Swish formula is then the out obtained from the GroupNorm formula.

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
      <td>x in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>gamma parameter in group normalization, which is gamma in the computation formula of `yOut`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>beta parameter in group normalization, which is beta in the computation formula of `yOut`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numGroups</td>
      <td>Attribute</td>
      <td>Divides dimension 1 of the input $x$ into group groups.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dataFormatOptional</td>
      <td>Attribute</td>
      <td>The current version supports only the input NCHW; character type on the host side.</td>
      <td>char*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td>Offset used to prevent division by 0, which is the eps value in the computation formulas of `yOut` and `rstdOut`.</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateSwish</td>
      <td>Attribute</td>
      <td>Represents whether Swish computation is supported.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>swishScale</td>
      <td>Attribute</td>
      <td>scale value used in the Swish computation.</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>Output</td>
      <td>Group normalization result.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>meanOut</td>
      <td>Output</td>
      <td>Mean of x after grouping.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstdOut</td>
      <td>Output</td>
      <td>Reciprocal of the standard deviation of x after grouping.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: the data types of all inputs and outputs do not support BFLOAT16.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_group_norm_swish.cpp](examples/test_aclnn_group_norm_swish.cpp) | Invokes the aclnnGroupNormSwish operator through the [aclnnGroupNormSwish](docs/aclnnGroupNormSwish_en.md) interface. |
| Graph mode | -  | Invokes the GroupNormSwish operator by building a graph through the [operator IR](op_graph/group_norm_swish_proto.h).         |
