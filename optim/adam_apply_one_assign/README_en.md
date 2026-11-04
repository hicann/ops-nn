# AdamApplyOneAssign

English | [简体中文](./README.md)

## Applicable Products

| Product                                              | Support |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR&950DT products</term>          |    √     |
| <term>Atlas A3 products</term>   |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>             |    √     |
| <term>Atlas inference products</term>                       |    ×     |
| <term>Atlas training products</term>                       |    √     |

## Function

- Operator function: performs one parameter update of the Adam optimizer, with 10 inputs and 3 outputs, and supports broadcast. The output port names input1, input2, and input3 are the same as the input names (GE inplace alias), and the framework aliases the output memory to the input memory to achieve in-place update; the output shapes are the same as the corresponding inputs respectively.

- Formula:

$$
out\_input1 = input1 × mul2\_x + input0² × mul3\_x
$$

$$
out\_input2 = mul0\_x × input2 + mul1\_x × input0
$$

$$
out\_input3 = input3 - out\_input2 / (sqrt(out\_input1) + add2\_y) × input4
$$

- Broadcast support: the shapes of all inputs must satisfy the broadcast relationship, the maximum shape after broadcasting must be the same as the shapes of inputs input1, input2, and input3, and the output shapes are the same as the corresponding inputs.

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
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
    <td>input0</td>
    <td>Input</td>
    <td>Input tensor input0 in the formula.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input1</td>
    <td>Input</td>
    <td>Input tensor input1 in the formula. It shares the GM address with the graph output port input1 (in-place update).</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input2</td>
    <td>Input</td>
    <td>Input tensor input2 in the formula. It shares the GM address with the graph output port input2 (in-place update).</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input3</td>
    <td>Input</td>
    <td>Input tensor input3 in the formula. It shares the GM address with the graph output port input3 (in-place update).</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input4</td>
    <td>Input</td>
    <td>Input tensor input4 in the formula.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mul0_x</td>
    <td>Input</td>
    <td>Input tensor mul0_x in the formula.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mul1_x</td>
    <td>Input</td>
    <td>Input tensor mul1_x in the formula.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mul2_x</td>
    <td>Input</td>
    <td>Input tensor mul2_x in the formula.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mul3_x</td>
    <td>Input</td>
    <td>Input tensor mul3_x in the formula.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>add2_y</td>
    <td>Input</td>
    <td>Input tensor add2_y in the formula.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input1</td>
    <td>Output</td>
    <td>Updated tensor, that is, out_input1 in the formula. It shares Device memory with the input input1 (inplace alias), and the shape/dtype are exactly the same as those of the input input1.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input2</td>
    <td>Output</td>
    <td>Updated tensor, that is, out_input2 in the formula. It shares Device memory with the input input2 (inplace alias), and the shape/dtype are exactly the same as those of the input input2.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>input3</td>
    <td>Output</td>
    <td>Updated tensor, that is, out_input3 in the formula. It shares Device memory with the input input3 (inplace alias), and the shape/dtype are exactly the same as those of the input input3.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- The maximum number of dimensions of the input shapes is 8.
- The data types of all inputs and outputs must be consistent.
- The outputs input1, input2, and input3 have the same names as the inputs input1, input2, and input3 (GE ports with the same name = inplace alias), the shapes are the same as the corresponding inputs respectively, and the caller must treat these three inputs as writable.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_adam_apply_one_assign](./examples/test_geir_adam_apply_one_assign.cpp)   | Calls the AdamApplyOneAssign operator by constructing a graph through the operator IR. |
