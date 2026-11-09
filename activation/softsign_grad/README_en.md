# SoftsignGrad

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

- Operator function: completes the backward gradient computation of the Softsign activation function. Given the upstream gradient gradients and the forward input features, it computes the output gradient element by element.
- Computation formula:

  $$
  output_i = \frac{gradients_i}{(1 + |features_i|)^2}
  $$

  where:
  - gradients: upstream gradient tensor
  - features: input feature tensor of the Softsign forward function
  - output: computed gradient output tensor

## Parameters

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">Parameter name</th>
      <th style="white-space: nowrap">Input/Output</th>
      <th style="white-space: nowrap">Description</th>
      <th style="white-space: nowrap">Data type</th>
      <th style="white-space: nowrap">Data format</th>
      <th style="white-space: nowrap">Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>gradients</td>
      <td>Input</td>
      <td>Upstream gradient tensor, corresponding to gradients in the formula. It supports 0 to 8 dimensional tensors, dynamic shapes, and empty tensors. Its shape must be exactly the same as the shape of features.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>×</td>
    </tr>
    <tr>
      <td>features</td>
      <td>Input</td>
      <td>Input feature tensor of the Softsign forward function, corresponding to features in the formula. It supports 0 to 8 dimensional tensors, dynamic shapes, and empty tensors. Its shape must be exactly the same as the shape of gradients.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>×</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Computed gradient output tensor, corresponding to output in the formula. Its shape is the same as the inputs, and its data type is consistent with the inputs.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>×</td>
    </tr>
  </tbody></table>

## Constraints

- The data types of gradients and features must be consistent, and only FLOAT, FLOAT16, and BFLOAT16 are supported.
- The shapes of gradients and features must be exactly the same, and broadcasting is not supported.
- Non-contiguous tensors (transposed or sliced views with strides, etc.) are not within the supported scope. The current version does not guarantee explicit interception or automatic conversion of non-contiguous inputs; callers should ensure that contiguous dense-layout tensors are passed in.
- Only 0 to 8 dimensional tensors are supported. Violations of the dimension count, format (ND only), or dtype constraints are rejected in a controlled manner at the graph compilation/tiling stage (the kernel is not executed).
- FP16/BF16 inputs are internally promoted to FP32 precision for the computation, and the results are converted back to the original precision.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode  | [test_geir_softsign_grad.cpp](examples/test_geir_softsign_grad.cpp) | Invokes the [SoftsignGrad operator](op_graph/softsign_grad_proto.h) in graph mode. |
