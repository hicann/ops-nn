# BinaryCrossEntropyGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: computes the gradient value of the binary cross entropy backpropagation.
- Formula:

  The formula of binary cross entropy is as follows:

  $$
  z_i = - weight_i \cdot (y_i \cdot ln(x_i) + (1 - y_i) \cdot ln(1- x_i))
  $$

  Here $x$ denotes the output of the previous layer of the network, that is, the forward prediction value, and $y$ denotes the label value of the sample. Compute the partial derivative of the binary cross entropy on $x$:

  $$
  \begin{aligned}
  \frac {\partial z}{\partial x} &= -weight \cdot \frac {\partial ((y_i \cdot ln(x_i) + (1-y_i) \cdot ln(1-x_i))}{\partial x} \\
  &= -weight \cdot (\frac {\partial (y \cdot ln(x))}{\partial x} + \frac {\partial ((1-y) \cdot ln(1-x))}{\partial x}) \\
  &= -weight \cdot (\frac {y}{x} - \frac {(1-y)}{1-x}) \\
  &= -weight \cdot \frac {y(1-x)-x(1-y)}{x(1-x)} \\
  &= -weight \cdot \frac {y-x}{x(1-x)} \\
  &= weight \cdot \frac {x-y}{x(1-x)}
  \end{aligned}
  $$

  $$
  out = grad\_output \cdot weight \cdot \frac {x-y}{x(1-x)} \\
  out = mean(grad\_input) \ if \ reduction = mean
  $$

## Parameters

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">Parameter name</th>
      <th style="white-space: nowrap">Input/Output/Attribute</th>
      <th style="white-space: nowrap">Description</th>
      <th style="white-space: nowrap">Data type</th>
      <th style="white-space: nowrap">Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Computation result of the previous layer in the forward pass of the network. The data type needs to be converted to the promotion type together with the other parameters.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Input</td>
      <td>Label value of the sample. The data type needs to be converted to the promotion type together with the other parameters, and the shape can be broadcast to the shape of x.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_output</td>
      <td>Input</td>
      <td>Gradient value of the previous step in the backpropagation of the network. The data type needs to be converted to the promotion type together with the other parameters, and the shape can be broadcast to the shape of x.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Weight of the result. The data type needs to be converted to the promotion type together with the other parameters, and the shape can be broadcast to the shape of x.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Stores the gradient computation result, with the same shape as x.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Attribute</td>
      <td>Reduce operation applied to the gradient computation result of the binary cross entropy backward.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>

  </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_binary_cross_entropy_backward.cpp](examples/arch35/test_aclnn_binary_cross_entropy_backward.cpp) | Calls the BinaryCrossEntropyGrad operator through the [aclnnBinaryCrossEntropyBackward](docs/aclnnBinaryCrossEntropyBackward_en.md) interface. |
