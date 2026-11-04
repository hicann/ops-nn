# SwishGrad

English | [简体中文](./README.md)


## Applicable Products

| Product | Support |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: the backpropagation of the Swish activation function, used to compute the gradient of the Swish activation function.
- Formula:

  - Formula of the Swish function

  $$
  s(x) = x*\sigma(\beta x)
  $$

  - Derivative implementation of the Swish function formula

  $$

  s^\prime(x)= \beta s(x)+\sigma(\beta x)(1-\beta s(x))= \sigma(\beta x)*(1+\beta x(1-\sigma(\beta x)))

  $$

  $$
  gradInput = gradOutput * s^\prime(x)
  $$

  $$
  \sigma(x) = {\frac{1} {1+{e}^{-x}}}
  $$

  Here $\sigma(x)$ is the Sigmoid function, $s(x)$ is the Swish function, and $s^\prime(x)$ is the derivative of the Swish function.

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
      <td>grad</td>
      <td>Input</td>
      <td>Denotes the gradient of the forward output of the Swish activation function, gradOutput in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Denotes the tensor used to compute the activation function, x in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Input</td>
      <td>Denotes the forward output result of the Swish activation function, σ(x) in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>Output</td>
      <td>Denotes the gradient value of the Swish forward input, that is, the result after differentiating the input.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Optional attribute</td>
      <td>Denotes an adjustable parameter, a scalar used to control the shape and slope of the Swish function, β in the formula.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface | [test_aclnn_swish_grad](examples/test_aclnn_swish_grad.cpp) | Calls the SwishGrad operator through the [aclnnSwishBackward](docs/aclnnSwishBackward_en.md) interface. |
| Graph mode | - | Calls the SwishGrad operator by building a graph through [operator IR](op_graph/swish_grad_proto.h). |
