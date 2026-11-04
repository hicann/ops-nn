# SigmoidGrad

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

- Description: Performs the backpropagation of [sigmoid](../sigmoid/docs/aclnnSigmoid&aclnnInplaceSigmoid_en.md), and computes the gradient input of sigmoid according to the backpropagation gradient and forward output of sigmoid.
- Formula:

  $$
  out = {\frac{1} {1+{e}^{-input}}}
  $$

  $$
  grad\_input = grad\_output * \sigma(x) * (1 - \sigma(x))
  $$

Here, $\sigma(x)$ is the forward output of the sigmoid function, and $\sigma(x)*(1-\sigma(x))$ is the derivative of the sigmoid function.

## Parameters

  <table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 230px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage instructions</th>
      <th>Data type</th>
      <th>Data format</th>
      <th>Dimensions (shape)</th>
      <th>Non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOutput</td>
      <td>Input</td>
      <td>Gradient of the loss function with respect to the sigmoid output, which is grad\_output in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape must be consistent with output and grad_input.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Input</td>
      <td>Output of the forward sigmoid, which is out in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape must be consistent with grad_output and grad_input.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>Output</td>
      <td>Gradient value of self, which is grad\_input in the formula.</td>
      <td>The data type and shape must be consistent with grad_output and output.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
   </tbody>
  </table>

 - The data types supported by SigmoidGrad operator registration are FLOAT16, BFLOAT16, and FLOAT.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                                                         |
| ---------------- | --------------------------- |----------------------------------------------------------------------------|
| aclnn API  | [test_aclnn_sigmoid_grad.cpp](examples/test_aclnn_sigmoid_grad.cpp) | Calls the SigmoidGrad operator through the [aclnnSigmoidBackward](docs/aclnnSigmoidBackward_en.md) API. |
| Graph mode | - | Calls the SigmoidGrad operator through graph construction with the [operator IR](op_graph/sigmoid_grad_proto.h). |
