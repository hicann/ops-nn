# FatreluMul

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |
|  <term>Kirin X90 processor series products</term>    |     √    |
|  <term>Kirin 9030 processor series products</term>    |     √    |

## Function

- Description:

  Splits the input tensor into left and right tensors along the last dimension: x1 and x2, performs the Threshold computation on the left x1, and multiplies the computation result by x2.

- Formula:

  Given the input tensor x whose last dimension has a length of 2d, perform the following computations:

  1. Split x into two parts:

     $$
     x_1 = x[..., :d], \quad x_2 = x[..., d:]
     $$
  2. Apply the Threshold activation function to x1, defined as follows:

     $$
     \text{Threshold}(x, \text{threshold}) =
        \begin{cases}
        0 & \text{if } x \leq \text{threshold} \\
        x & \text{if } x > \text{threshold}
        \end{cases}
     $$

     Therefore, the computation is:

     $$
     x_1 = \text{Threshold}(x_1, \text{threshold})
     $$
  3. The final output is the element-wise product of x1 and x2:

     $$
     \text{out} = x_1 \times x_2
     $$

## Parameters

<table style="undefined;table-layout: fixed; width: 891px"><colgroup>
  <col style="width: 121px">
  <col style="width: 144px">
  <col style="width: 253px">
  <col style="width: 257px">
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
      <td>x</td>
      <td>Input</td>
      <td>The input x in the formulas.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>Input</td>
      <td><ul><li>The input threshold in the formulas.</li><li>threshold does not support NaN and Inf.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The out in the formulas.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

In typical scenarios, the tail axis is a multiple of 16. When the tail axis is not 32-byte aligned, using the concatenation logic of small operators is recommended.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_fatrelu_mul](./examples/arch35/test_aclnn_fatrelu_mul.cpp) | Invokes the FatreluMul operator through the [aclnnFatreluMul](./docs/aclnnFatreluMul_en.md) interface.    |
| Graph-mode invocation | -   | Builds the graph and invokes the FatreluMul operator through the [operator IR](./op_graph/fatrelu_mul_proto.h). |
