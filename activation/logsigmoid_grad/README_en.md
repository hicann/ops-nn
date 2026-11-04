# LogSigmoidGrad

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

- Operator function: This is the backpropagation of [LogSigmoid](../log_sigmoid/docs/aclnnLogSigmoid_en.md). It computes the gradient input based on the gradient propagated from the previous layer and the forward input of LogSigmoid.

- Formula:

  $$
  \sigma(x) = LogSigmoid(x) = \log(\frac{1}{1+e^{-x}})
  $$

  $$
  gradInput = gradOutput * \sigma^\prime(x) = gradOutput * \frac{1}{1+e^{x}}
  $$

  Where $x$ is the forward input of the LogSigmoid function, $\sigma(x)$ is the LogSigmoid function, and $\sigma^\prime(x)$ is the derivative of the LogSigmoid function.

- **Parameters:**

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
      <td>Represents the gradient of the forward output of the LogSigmoid activation function, gradOutput in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>features</td>
      <td>Input</td>
      <td>Input parameter for the backward computation, x in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops</td>
      <td>Output</td>
      <td>Output parameter, gradInput in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

- The data types of inputs and outputs remain the same.

## Invocation Example

| Invocation method   | Sample code                                                                         | Description                                                                               |
| ---------------- |------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| aclnn interface  | [test_aclnn_logsigmoid_grad.cpp](examples/test_aclnn_logsigmoid_grad.cpp) | Invokes the LogSigmoidGrad operator through the [aclnnLogSigmoidBackward](docs/aclnnLogSigmoidBackward_en.md) interface. |
