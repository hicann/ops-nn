# GeluMul

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term>    |     √    |
|  <term>Kirin 9030 processor series products</term>    |     √    |

## Function

- Description:

  Split the input Tensor into the left and right Tensors x1 and x2 along the last dimension, perform the GELU computation on the left x1, and multiply the computation result by x2.

- Formula:

  Given the input tensor `input` whose last dimension has a length of `2d`, the function `GeluMul` performs the following computation:

  1. Split `input` into two parts:

     $$
     x_1 = \text{input}[..., :d], \quad x_2 = \text{input}[..., d:]
     $$
  2. Apply the GELU activation function to x1. The formula of the "tanh" mode is as follows:

     $$
     \text{GELU}(x) = 0.5 \cdot x \cdot \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \cdot \left( x + 0.044715 x^3 \right) \right) \right)
     $$

     The formula of the erf mode corresponding to "none" is as follows:

     $$
     \text{GELU}(x) = 0.5 \cdot x \left( 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right)
     $$

     Therefore, the operator computes:

     $$
     x_1 = \text{GELU}(x_1)
     $$
  3. The final output is the element-wise product of x1 and x2:

     $$
     \text{out} = x_1 \times x_2
     $$

## Parameters

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
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
      <td>The input input in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>approximateOptional</td>
      <td>Optional attribute</td>
      <td>The mode of the GELU computation. Only "none" and "tanh" are supported, corresponding to the erf mode and the tanh mode of GELU respectively. When a null pointer is passed in, the value is "none".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The out in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

In a typical scenario, the trailing axis is a multiple of 16. When the trailing axis is not 32B aligned, it is recommended to use the small-operator concatenation logic.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_gelu_mul](./examples/test_aclnn_gelu_mul.cpp) | Invoke the GeluMul operator through the [aclnnGeluMul](./docs/aclnnGeluMul_en.md) interface.    |
| Graph-mode invocation | -   | Invoke the GeluMul operator by constructing a graph through the [operator IR](./op_graph/gelu_mul_proto.h). |
