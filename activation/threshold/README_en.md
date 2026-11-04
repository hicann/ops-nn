# Threshold

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Performs the threshold operation on the input x. When the elements in x are greater than threshold, the operator returns 1. Otherwise, it returns 0.

- Formula:

$$
y = (x > threshold) \; ? \; 1 : 0
$$

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>Input parameter for the threshold computation, which is x in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter after the threshold computation, which is y in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>Attribute</td>
      <td>Threshold. Elements in x that are greater than this value output 1, and otherwise output 0. Default value: 0.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The input and output are tensors of arbitrary dimensions (ND format).
- The data types of the input and output must be consistent.
- The output values are discrete values (0.0 or 1.0), which are bit-exact consistent with the Golden result.
- A NaN input outputs 0.0 (NaN compared with any value is always False).
- A +Inf input outputs 1.0 (+Inf > threshold is always True).
- A -Inf input outputs 0.0 (-Inf > threshold is always False).

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| Graph-mode invocation | [test_geir_threshold](./examples/test_geir_threshold.cpp)   | Calls the Threshold operator through graph construction with the <a href="op_graph/threshold_relu_proto.h">operator IR</a>. |
