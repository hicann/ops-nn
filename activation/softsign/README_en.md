# Softsign

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |
| Atlas 200I/500 A2 inference products | √ |
| Atlas inference products | √ |
| Atlas training products | √ |

## Function

- Operator function: completes the Softsign activation function computation, computing the softsign value elementwise for each element of the input tensor.
- Formula:

  $$
  y = \frac{x}{1 + |x|}
  $$

  In the formula, $x$ is the input tensor, $y$ is the output tensor, and the output value range is $(-1, 1)$.

- Numerical properties: the formula is naturally numerically stable. The denominator $1 + |x| \geq 1$, there is no division-by-zero risk, and no epsilon protection is needed.

## Parameters

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>The input x in the formula, a tensor of any shape.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>The output y in the formula, with a shape and dtype exactly matching input x, and a value range of (-1, 1).</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- Input dimension limit: the input tensor has no more than 8 dimensions, and 0-dimensional scalar inputs are supported.
- Data type limit: only FLOAT, FLOAT16, and BFLOAT16 are supported; DOUBLE is not supported.
- FP16/BF16 precision description: FP16 and BFLOAT16 types perform intermediate computation internally by casting to FP32 and then casting back to the original type, to guarantee computation precision.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td> Graph-mode invocation </td>
    <td><a href="./examples/arch35/test_geir_softsign.cpp">test_geir_softsign</a></td>
    <td> Calls the Softsign operator by building the graph through the [operator IR](./op_graph/softsign_proto.h). </td>
  </tr>
</tbody></table>
