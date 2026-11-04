# HardSigmoidGrad

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

- Operator function: This is the backward operator of [HardSigmoid](../hard_sigmoid/README.md). It computes the backpropagation gradient grad_input.

- Formula:

  $$
  HardsigmoidBackward(self, grad\_output)=
  \begin{cases}
  &grad\_output \ast \alpha, &if(0 < \alpha \ast self + \beta < 1) \\
  &0, &otherwise
  \end{cases}
  $$

## Parameters

  <table style="undefined;table-layout: fixed; width: 800px"><colgroup>
  <col style="width: 110px">
  <col style="width: 130px">
  <col style="width: 300px">
  <col style="width: 180px">
  <col style="width: 80px">
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
      <td>Gradient output by the previous step during backpropagation, used as the input of this backward operator, grad_output in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_x</td>
      <td>Input</td>
      <td>Represents the input tensor of the HardSigmoid forward operator, self in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Backpropagation gradient output, HardsigmoidBackward(self, grad_output) in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
   </tr>
  </tbody>
  </table>

  - In the formula, $\alpha$ is fixed to $1/6$ and $\beta$ is fixed to $0.5$.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code                                                                         | Description                                                                                 |
| ---------------- |------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| aclnn interface  | [test_aclnn_hard_sigmoid_grad.cpp](examples/test_aclnn_hard_sigmoid_grad.cpp) | Invokes the HardSigmoidGrad operator through the [aclnnHardsigmoidBackward](docs/aclnnHardsigmoidBackward_en.md) interface. |
