# GeluGrad

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

## Function

- Description: backward propagation operator of the [Gelu](../gelu/README.md) activation function, used to compute the gradient of the Gelu activation function.
- Formula:
  - Forward formula

  $$
  out=GELU(self)=self × Φ(self)=0.5 * self * (1 + tanh( \sqrt{2 / \pi} * (self + 0.044715 * self^{3})))
  $$

  - Backward gradient formula

  $$
   out = \frac{d(\text{GELU})}{dx} = dy \cdot \left[ \underbrace{0.5(1 + \tanh(\text{inner}))}_{\text{left\_derivative}}+ \underbrace{0.5x\cdot (1-\tanh^2(\text{inner})) \cdot \beta(1+3\cdot 0.044715x^2)}_{\text{right\_derivative}}\right]

  $$

  - In this formula:

  $$

   \beta = \sqrt{\frac{2}{\pi}},\quad \text{inner}= \beta \left(x+0.044715x^3 \right)
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 919px"><colgroup>
  <col style="width: 130px">
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
      <td>dy</td>
      <td>Input</td>
      <td>This is the gradient of the loss function with respect to the output of GELU, dy in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>This is the input of the function, x in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Input</td>
      <td>Output of the GELU function, that is, GELU(x).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>Output</td>
      <td>Denotes the output of the gelu_grad function, corresponding to out in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_gelu_backward](./examples/test_aclnn_gelu_backward.cpp) | Invokes the GeluGrad operator through the [aclnnGeluBackward](./docs/aclnnGeluBackward_en.md) API.    |
