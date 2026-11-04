# LambNextRight

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | × |
|  <term>Kirin 9030 processor series products</term> | × |

## Function

- Operator function: A graph fusion operator of the BERT LAMB optimizer: computes the second moment next_v and its bias-corrected denominator sqrt(v_unbiased)+epsilon.

- Formula:

  $y1 = input\_mul2 \times mul2\_x + input\_square^2 \times mul3\_x\quad(next\_v)$

  $y2 = \sqrt{y1 \times truediv1\_recip} + add2\_y$

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
      <th>Input/Output</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input_square</td>
      <td>Input</td>
      <td>Supports empty tensors. input_square in the formula (gradient g), the main tensor; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul2</td>
      <td>Input</td>
      <td>Supports empty tensors. input_mul2 in the formula (second moment v), the main tensor; the shape must satisfy the broadcast relationship with other inputs, and the broadcast result of all inputs determines the shape of each output.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul2_x</td>
      <td>Input</td>
      <td>Supports empty tensors. mul2_x in the formula (beta2); the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul3_x</td>
      <td>Input</td>
      <td>Supports empty tensors. mul3_x in the formula (1-beta2); the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>truediv1_recip</td>
      <td>Input</td>
      <td>Supports empty tensors. truediv1_recip in the formula (reciprocal of the bias-correction denominator); the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>add2_y</td>
      <td>Input</td>
      <td>Supports empty tensors. add2_y in the formula (epsilon); the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>Supports empty tensors. y1 in the formula (next_v); the shape is the broadcast result of all inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Supports empty tensors. y2 in the formula (bias-correction denominator); the shape is the broadcast result of all inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The shapes of all inputs must pairwise satisfy the broadcast rules, and the output shape is the broadcast result of all inputs.
- The numbers of dimensions of all inputs and outputs are 0~8. When all inputs are 0-dimensional, the shapes of y1 and y2 are (1,).
- The data types of all inputs must be the same, either all FLOAT16 or all FLOAT.

## Invocation Description

| Invocation method | Invocation sample | Description |
|---|---|---|
| Graph-mode invocation | [test_geir_lamb_next_right](./examples/test_geir_lamb_next_right.cpp) | Calls the LambNextRight operator by building a graph through the operator IR. |
