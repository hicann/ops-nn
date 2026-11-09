# Heaviside

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>kirin X90 processor series products</term>    |     √    |
|  <term>Kirin 9030 processor series products</term>    |     √    |

## Function

- Description:
  Computes the Heaviside step function of each element in the input input, serving as the activation function of the model.

- Formula:

  $$
  \text{Heaviside}(\text{input},   \text{values}) =
  \begin{cases}
  0, & \text{if input} < 0 \\
  \text{values}, & \text{if input} = 0 \\
  1, & \text{if input} > 0
  \end{cases}
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 813px"><colgroup>
  <col style="width: 103px">
  <col style="width: 144px">
  <col style="width: 194px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>input</td>
      <td>Input</td>
      <td>Input input in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>Input</td>
      <td>Input values in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Heaviside in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- The data types of input, values, and out must be consistent.
- input supports 1~8 dims. When values is a single-element tensor (scalar), input additionally supports 0 dims (scalar).
- values supports only the following two forms:
  - A single-element tensor (with a total element count of 1, processed as a scalar);
  - The exact same shape as input, with the total element counts of both input and values greater than 0 (graph mode does not support empty tensors with a total element count of 0).
- In graph mode (operator IR), the kernel does not perform broadcast internally and reads values only as a scalar or at the same linear offset. Therefore, a general broadcast relationship between input and values is not supported (for example, input is a 0-dim scalar while values is a multi-dim tensor).

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_heaviside](./examples/test_aclnn_heaviside.cpp) | Calls the Heaviside operator through the [aclnnHeaviside](./docs/aclnnHeaviside_en.md) interface.    |
| Graph-mode invocation | -   | Calls the Heaviside operator by constructing a graph through the [operator IR](./op_graph/heaviside_proto.h). |
