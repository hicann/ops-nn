# LambApplyWeightAssign

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

- Operator function: completes the trust ratio weight update of the LAMB optimizer. It computes the trust ratio based on the weight norm input0 and the gradient norm input1, then scales the update with the learning rate input2 and updates the parameter input_param in place.

- Formula:

  $ratio = where(input0>0,\ where(input1>0,\ input0/input1,\ 1),\ 1)$

  $input\_param = input\_param - ratio \times (input3 \times input2)$

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
      <td>input0</td>
      <td>Input</td>
      <td>Supports empty tensors. It is input0 (the weight norm) in the formula. The shape supports 0 to 8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input1</td>
      <td>Input</td>
      <td>Supports empty tensors. It is input1 (the gradient norm) in the formula. The shape supports 0 to 8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input2</td>
      <td>Input</td>
      <td>Supports empty tensors. It is input2 (the learning rate) in the formula. The shape supports 0 to 8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input3</td>
      <td>Input</td>
      <td>Supports empty tensors. It is input3 (update) in the formula. The shape supports 0 to 8 dimensions and must satisfy the broadcast rules with other inputs.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_param</td>
      <td>Input</td>
      <td>Supports empty tensors. It is input_param (the parameter) in the formula. input_param is the <b>in-place update</b> output. Its shape must be equal to the broadcast result of all inputs (that is, all other inputs must be able to broadcast into input_param).</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_param</td>
      <td>Output</td>
      <td>Supports empty tensors. The updated input_param (in-place update). The shape is the broadcast result of all inputs and equals the shape of input_param.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The shapes of all inputs must pairwise satisfy the broadcast rules. `input_param` is the output of the in-place update, so the broadcast result of all inputs must exactly equal the shape of `input_param`. Otherwise, the in-place write-back goes out of bounds.
- The number of dimensions of all inputs and outputs is 0 to 8. When all inputs are 0-dimensional, the shape of the output is (1,).
- The data types of all inputs must be consistent, all FLOAT16 or all FLOAT.

## Invocation Description

| Invocation method | Invocation sample | Description |
|---|---|---|
| Graph-mode invocation | [test_geir_lamb_apply_weight_assign](./examples/test_geir_lamb_apply_weight_assign.cpp) | Invokes the LambApplyWeightAssign operator through graph construction using the operator IR. |
