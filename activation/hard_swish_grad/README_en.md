# HardSwishGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Backpropagation of the HardSwish activation function, completing the gradient computation of the tensor self.
- Formula:

  $$
  out_{i} = gradOutput_{i} \times gradSelf_{i}
  $$

  The calculation formula of gradSelf is:

  $$
  gradSelf_{i} = \begin{cases}
  0, & self_{i} \lt -3, \\
  self_{i} / 3 + 0.5, & -3 \le self_{i} \le 3, \\
  1, & self_{i} \gt 3
  \end{cases}
  $$

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
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>grad_output</td>
      <td>Input</td>
      <td>Denotes the gradient of the forward output of the HardSwish activation function, gradOutput in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Denotes the tensor used to compute the activation function, self in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>Output</td>
      <td>Denotes the gradient value of the HardSwish forward input, that is, the result after differentiating the input.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface | [test_aclnn_hard_swish_backward](examples/test_aclnn_hard_swish_backward.cpp) | Calls the HardSwishGrad operator through the [aclnnHardswishBackward](docs/aclnnHardswishBackward_en.md) interface. |
