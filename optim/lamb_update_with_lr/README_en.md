# LambUpdateWithLr

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
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

- Operator function: A graph fusion operator of the BERT LAMB optimizer (trust ratio weight update, with clipping): computes the trust ratio, clips it with minimum_y/greater_y as the upper and lower bounds, and updates parameter input_sub.

- Formula:

  $ratio = where(input\_greater1>greater\_y,\ input\_greater\_realdiv/input\_realdiv,\ select\_e)$

  $clip = max(min(ratio,\ minimum\_y),\ greater\_y)$

  $y = input\_sub - clip \times input\_mul0 \times input\_mul1$

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
      <td>input_greater1</td>
      <td>Input</td>
      <td>Supports empty tensors. input_greater1 in the formula (weight norm); the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_greater_realdiv</td>
      <td>Input</td>
      <td>Supports empty tensors. input_greater_realdiv in the formula (numerator of the trust ratio); the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_realdiv</td>
      <td>Input</td>
      <td>Supports empty tensors. input_realdiv in the formula (denominator of the trust ratio); the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul0</td>
      <td>Input</td>
      <td>Supports empty tensors. input_mul0 in the formula (learning rate); the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul1</td>
      <td>Input</td>
      <td>Supports empty tensors. input_mul1 in the formula (update), the main tensor; the shape must satisfy the broadcast relationship with other inputs, and the broadcast result of all inputs determines the shape of output y.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_sub</td>
      <td>Input</td>
      <td>Supports empty tensors. input_sub in the formula (parameters), the main tensor; the shape must satisfy the broadcast relationship with other inputs, and the broadcast result of all inputs determines the shape of output y.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>greater_y</td>
      <td>Input</td>
      <td>Supports empty tensors. greater_y in the formula (threshold); the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>select_e</td>
      <td>Input</td>
      <td>Supports empty tensors. select_e in the formula (fallback value); the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>minimum_y</td>
      <td>Input</td>
      <td>Supports empty tensors. minimum_y in the formula (clipping upper bound); the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Supports empty tensors. y in the formula (updated parameters); the shape is the broadcast result of all inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The shapes of all inputs must pairwise satisfy the broadcast rules, and the output shape is the broadcast result of all inputs.
- The numbers of dimensions of all inputs and outputs are 0~8. When all inputs are 0-dimensional, the output shape is (1,).
- The data types of all inputs must be the same, either all FLOAT16 or all FLOAT.

## Invocation Description

| Invocation method | Invocation sample | Description |
|---|---|---|
| Graph-mode invocation | [test_geir_lamb_update_with_lr](./examples/test_geir_lamb_update_with_lr.cpp) | Calls the LambUpdateWithLr operator by building a graph through the operator IR. |
