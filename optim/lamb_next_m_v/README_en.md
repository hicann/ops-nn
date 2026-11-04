# LambNextMV

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | × |
|  <term>Kirin 9030 processor series products</term> | × |

## Function

- Operator function: a BERT LAMB optimizer graph fusion operator. Based on the intermediate quantities already computed upstream (g^2, the first/second moments, the bias-correction denominators, and so on), it performs the update computation after the Adam moment update and bias correction.

- Formula:

  $y3 = input\_mul2 \times mul2\_x + input\_mul3 \times mul3\_sub1\quad(next\_v)$

  $y2 = input\_mul0 \times mul0\_x + input\_mul1 \times mul1\_sub\quad(next\_m)$

  $y1 = input\_mul4 \times mul4\_x + \frac{y2/input\_realdiv0}{\sqrt{y3/input\_realdiv1 + add2\_y}}$

  $y4 = \frac{y2/input\_realdiv0}{\sqrt{y3/input\_realdiv1} + add2\_y}$

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
      <td>input_mul3</td>
      <td>Input</td>
      <td>Supports empty tensors. It is input_mul3 (g^2) in the formula, the main tensor. The shape supports 0 to 8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul2</td>
      <td>Input</td>
      <td>Supports empty tensors. It is input_mul2 (the second moment v) in the formula, the main tensor.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_realdiv1</td>
      <td>Input</td>
      <td>Supports empty tensors. It is input_realdiv1 (1-beta2^t) in the formula, the main tensor.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul1</td>
      <td>Input</td>
      <td>Supports empty tensors. It is input_mul1 (the gradient g) in the formula, the main tensor.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul0</td>
      <td>Input</td>
      <td>Supports empty tensors. It is input_mul0 (the first moment m) in the formula, the main tensor. The shape must satisfy the broadcast relationship with other inputs. The broadcast result of all inputs determines the shapes of the outputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_realdiv0</td>
      <td>Input</td>
      <td>Supports empty tensors. It is input_realdiv0 (1-beta1^t) in the formula, the main tensor.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul4</td>
      <td>Input</td>
      <td>Supports empty tensors. It is input_mul4 (the parameter param) in the formula, the main tensor.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul0_x</td>
      <td>Input</td>
      <td>Supports empty tensors. It is mul0_x (beta1) in the formula. The shape supports 0 to 8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul1_sub</td>
      <td>Input</td>
      <td>Supports empty tensors. It is mul1_sub (1-beta1) in the formula. The shape supports 0 to 8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul2_x</td>
      <td>Input</td>
      <td>Supports empty tensors. It is mul2_x (beta2) in the formula. The shape supports 0 to 8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul3_sub1</td>
      <td>Input</td>
      <td>Supports empty tensors. It is mul3_sub1 (1-beta2) in the formula. The shape supports 0 to 8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul4_x</td>
      <td>Input</td>
      <td>Supports empty tensors. It is mul4_x (the weight decay coefficient) in the formula. The shape supports 0 to 8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>add2_y</td>
      <td>Input</td>
      <td>Supports empty tensors. It is add2_y (epsilon) in the formula. The shape supports 0 to 8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>Supports empty tensors. It is y1 (update) in the formula. The shape is the broadcast result of all inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Supports empty tensors. It is y2 (next_m) in the formula. The shape is the broadcast result of all inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y3</td>
      <td>Output</td>
      <td>Supports empty tensors. It is y3 (next_v) in the formula. The shape is the broadcast result of all inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y4</td>
      <td>Output</td>
      <td>Supports empty tensors. It is y4 in the formula. The shape is the broadcast result of all inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The shapes of all inputs must pairwise satisfy the broadcast rules, and the output shape is the broadcast result of all inputs.
- The number of dimensions of all inputs and outputs is 0 to 8. When all inputs are 0-dimensional, the shapes of y1, y2, y3, and y4 are (1,).
- The data types of all inputs must be consistent, all FLOAT16 or all FLOAT.
- The shapes of the outputs y1/y2/y3/y4 are all the broadcast result of all inputs.

## Invocation Description

| Invocation method | Invocation sample | Description |
|---|---|---|
| Graph-mode invocation | [test_geir_lamb_next_m_v](./examples/test_geir_lamb_next_m_v.cpp) | Invokes the LambNextMV operator through graph construction using the operator IR. |
