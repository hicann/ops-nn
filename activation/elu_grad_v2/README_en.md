# EluGradV2

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

- Operator function: Backward computation of the activation function. It outputs the gradient of the forward input of the ELU activation function.

- Formula:

  - When is_result is True:

      $$
      gradInput = gradOutput *
      \begin{cases}
      scale, \quad x > 0\\
      inputScale \ast (x + \alpha \ast scale),  \quad x \leq 0
      \end{cases}
      $$

  - When is_result is False:

      $$
      gradInput = gradOutput *
      \begin{cases}
      scale, \quad x > 0\\
      inputScale \ast \alpha \ast scale \ast exp(x \ast inputScale), \quad x \leq 0
      \end{cases}
      $$

## Parameters

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>grads</td>
      <td>Input</td>
      <td>gradOutput in the formula</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activations</td>
      <td>Input</td>
      <td>x in the formula</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Activation coefficient of the ELU activation function, alpha in the formula. If is_result is true, alpha must be greater than or equal to 0. The data type must be a data type that can be converted to FLOAT.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Scale coefficient of the ELU activation function, scale in the formula.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_scale</td>
      <td>Input</td>
      <td>Scale coefficient of the input of the ELU activation function, inputScale in the formula. The data type must be a data type that can be converted to FLOAT.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_result</td>
      <td>Input</td>
      <td>Whether the input passed to the ELU backward computation is the output of the ELU forward computation.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor gradInput in the formula.</td>
      <td>Same as grads</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface | [test_aclnn_elu_backward](examples/test_aclnn_elu_backward.cpp) | Calls the operator through the [aclnnEluBackward](docs/aclnnEluBackward_en.md) interface. |
