# LambUpdateWithLrV2

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

- Operator function: Graph fusion operator of the BERT LAMB optimizer (trust ratio weight update, without clipping). The operator computes the trust ratio with the threshold greater_y and the fallback value select_e, and then updates the parameter x5 with the learning rate x3.

- Formula:

  $ratio = where(x1>greater\_y,\ where(x2>greater\_y,\ x1/x2,\ select\_e),\ select\_e)$

  $y = x5 - x3 \times ratio \times x4$

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
      <td>x1</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is x1 (weight norm) in the formulas; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is x2 (gradient norm) in the formulas; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is x3 (learning rate) in the formulas; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x4</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is x4 (update) in the formulas; it is the main tensor, its shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x5</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is x5 (parameter) in the formulas; it is the main tensor, its shape must satisfy the broadcast relationship with other inputs, and the broadcast result of all inputs determines the shape of the output y.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>greater_y</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is greater_y (threshold) in the formulas; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>select_e</td>
      <td>Input</td>
      <td>Empty tensors are supported. It is select_e (fallback value) in the formulas; the shape supports 0-8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Empty tensors are supported. It is y (updated parameter) in the formulas; the shape is the broadcast result of all inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The shapes of all inputs must pairwise satisfy the broadcast rules, and the output shape is the broadcast result of all inputs.
- The number of dimensions of all inputs and outputs is 0~8. When all inputs are 0-dimensional, the shape of the output is (1,).
- The data types of all inputs must be the same, all FLOAT16 or all FLOAT.

## Invocation Description

| Invocation method | Invocation sample | Description |
|---|---|---|
| Graph-mode invocation | [test_geir_lamb_update_with_lr_v2](./examples/test_geir_lamb_update_with_lr_v2.cpp) | Builds the graph and invokes the LambUpdateWithLrV2 operator through the operator IR. |
