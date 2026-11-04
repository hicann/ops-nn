# ApplyProximalGradientDescent

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

- Operator function: Performs one-step update of Proximal Gradient Descent with L1/L2 regularization. Given the weight var to be updated, the learning rate alpha, the L1 regularization coefficient l1, the L2 regularization coefficient l2, and the gradient delta, the output is computed element-wise.
- Reference API: TensorFlow `tf.raw_ops.ResourceApplyProximalGradientDescent`.
- Formula:

  $$
  prox\_v = var - alpha \times delta
  $$

  $$
  varOut = \dfrac{\operatorname{sign}(prox\_v)}{1 + alpha \times l2} \times \max\bigl(|prox\_v| - alpha \times l1,\ 0\bigr)
  $$

  where `sign(0) = 0`, consistent with the TensorFlow semantics.

  Degenerate cases:

  | Condition            | Equivalent formula                                       |
  | --------------- | ---------------------------------------------- |
  | `l1 = 0, l2 = 0` | `varOut = var - alpha * delta` (standard SGD)      |
  | `l1 = 0`         | `varOut = (var - alpha * delta) / (1 + alpha * l2)` |

## Parameters

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">Parameter name</th>
      <th style="white-space: nowrap">Input/Output</th>
      <th style="white-space: nowrap">Description</th>
      <th style="white-space: nowrap">Data type</th>
      <th style="white-space: nowrap">Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>var</td>
      <td>Input</td>
      <td>Weight tensor to be updated. 1-8 dimensional tensors are supported, and non-contiguous tensors are supported. The shape must be exactly the same as the shape of delta.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>NC1HWC0, C1HWNCoC0, ND, FRACTAL_Z</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Learning rate scalar tensor (0-D or shape=[1], non-negative).</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l1</td>
      <td>Input</td>
      <td>L1 regularization coefficient scalar tensor (0-D or shape=[1], non-negative).</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l2</td>
      <td>Input</td>
      <td>L2 regularization coefficient scalar tensor (0-D or shape=[1], non-negative).</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>delta</td>
      <td>Input</td>
      <td>Gradient tensor, with the shape exactly the same as var. Non-contiguous tensors are supported.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>NC1HWC0, C1HWNCoC0, ND, FRACTAL_Z</td>
    </tr>
    <tr>
      <td>var (output)</td>
      <td>Output</td>
      <td>Output tensor, with the shape/dtype/format the same as the input var.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>NC1HWC0, C1HWNCoC0, ND, FRACTAL_Z</td>
    </tr>
  </tbody></table>

## Constraints

- The data types of all inputs must be the same, and only FLOAT16 and FLOAT32 are supported.
- The shapes of var and delta must be exactly the same, and broadcasting is not supported.
- alpha / l1 / l2 must be scalar tensors of 0-D or shape=[1].
- FP16 inputs are internally promoted to FP32 precision for computation, and the results are converted back to the original precision.
- Deterministic implementation by default.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode  | [test_geir_apply_proximal_gradient_descent.cpp](examples/test_geir_apply_proximal_gradient_descent.cpp) | Invoke the ApplyProximalGradientDescent operator in graph mode. |
