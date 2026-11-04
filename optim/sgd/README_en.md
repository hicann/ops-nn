# SGD

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: Updates weights according to the Stochastic Gradient Descent (SGD) algorithm with momentum,
  and updates the momentum accumulator and the first-step flag in place.
- Formula:

  Let $d$ be `dampening`, $wd$ be `weight_decay`, $lr$ be `learning_rate[0]`,
  and $m$ be `momentum[0]`. First compute:

  $$
  grad =
  \begin{cases}
  gradient + parameters \times wd, & wd \neq 0 \\
  gradient, & wd = 0
  \end{cases}
  $$

  $$
  accum_t = accum \times m + grad
  $$

  When $d \neq 0$:

  $$
  accum_t = accum_t - grad \times (1 - stat) \times d
  $$

  The weights are updated as follows:

  $$
  parameters_{out} =
  \begin{cases}
  parameters - (grad + accum_t \times m) \times lr, & nesterov = true \\
  parameters - accum_t \times lr, & nesterov = false
  \end{cases}
  $$

  When $m \neq 0$, `accum` is updated to `accum_t` and `stat` is updated to 0. When $m=0$,
  `accum` and `stat` keep their input values.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| parameters | Input | Weights to be updated. Must be a writable tensor. | FLOAT, FLOAT16, BFLOAT16 | NC1HWC0, NDC1HWC0, ND, FRACTAL_Z, FRACTAL_Z_3D |
| gradient | Input | Gradient. The shape, data type, and data format are the same as those of `parameters`. | FLOAT, FLOAT16, BFLOAT16 | NC1HWC0, NDC1HWC0, ND, FRACTAL_Z, FRACTAL_Z_3D |
| learning_rate | Input | Learning rate. The number of elements must be 1, and the data type is the same as that of `parameters`. | FLOAT, FLOAT16, BFLOAT16 | ND |
| accum | Input | Momentum accumulator. Must be a writable tensor. The shape, data type, and data format are the same as those of `parameters`. | FLOAT, FLOAT16, BFLOAT16 | NC1HWC0, NDC1HWC0, ND, FRACTAL_Z, FRACTAL_Z_3D |
| momentum | Input | Momentum factor. The number of elements must be 1, and the data type is the same as that of `parameters`. | FLOAT, FLOAT16, BFLOAT16 | ND |
| stat | Input | Element-wise first-step flag. A value of 1 indicates that dampening is not applied. Must be a writable tensor. The shape, data type, and data format are the same as those of `parameters`. | FLOAT, FLOAT16, BFLOAT16 | NC1HWC0, NDC1HWC0, ND, FRACTAL_Z, FRACTAL_Z_3D |
| parameters | Output | Updated weights. The shape, data type, and data format are the same as the input `parameters`. | FLOAT, FLOAT16, BFLOAT16 | NC1HWC0, NDC1HWC0, ND, FRACTAL_Z, FRACTAL_Z_3D |
| dampening | Optional attribute | Momentum dampening coefficient. Default value: 0.0. When `nesterov=true`, the value must be 0. | FLOAT | - |
| weight_decay | Optional attribute | Weight decay coefficient. Default value: 0.0. The value must be greater than or equal to 0. | FLOAT | - |
| nesterov | Optional attribute | Whether to enable Nesterov momentum. Default value: false. | BOOL | - |

## Constraints

- Only one `parameters` output is declared on the graph, but `parameters`, `accum`, and `stat` may all be updated in place.
  The caller must pass these three inputs as writable tensors. The weight update value should be read from the `parameters` output.
- `momentum=0` only prevents `accum` and `stat` from being written back, and does not skip the momentum multiplication in the weight update. When `accum` contains Inf, the NaN produced by `0×Inf` propagates to the weight result.
- <term>Ascend 950PR&950DT products</term>:
  - Data types: each input and output supports FLOAT, FLOAT16, and BFLOAT16.
  - Data formats: each input and output uses ND for both static shapes and dynamic shapes.
  - GE graph inputs provide data in the ND contiguous layout; non-contiguous views such as transposes and slices must be converted to contiguous tensors by the caller or the framework in advance.
  - Dynamic shape capability: dynamic shapes and dynamic ranks are supported.
  - Input shape constraint: the rank of `parameters` is 1 to 8. Empty tensors are not supported.
  - Intermediate computation precision: FLOAT is used for intermediate computation. FLOAT16 and BFLOAT16 results are converted back to the input data type with round-to-nearest-even.
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
  - Data types: each input and output supports FLOAT, FLOAT16, and BFLOAT16.
  - Data formats for static shapes: `parameters`, `gradient`, `accum`, `stat`, and the output support NC1HWC0, NDC1HWC0, ND, FRACTAL_Z, and FRACTAL_Z_3D. `learning_rate` and `momentum` are ND.
  - Dynamic shape capability: dynamic shapes are not supported.
  - Input shape constraint: the rank of `parameters` is 1 to 8. Empty tensors are not supported.
- <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>:
  - Data types: each input and output supports FLOAT and FLOAT16.
  - Data formats for static shapes: `parameters`, `gradient`, `accum`, `stat`, and the output support NC1HWC0, NDC1HWC0, ND, FRACTAL_Z, and FRACTAL_Z_3D. `learning_rate` and `momentum` are ND.
  - Dynamic shape capability: dynamic shapes are not supported.
  - Input shape constraint: the rank of `parameters` is 1 to 8. Empty tensors are not supported.

## Invocation Description

This operator does not provide an aclnn single-operator API and is invoked in the graph mode of the Graph Engine (GE).

| Invocation method | Sample code | Description |
| --- | --- | --- |
| GE graph mode | [test_geir_sgd.cpp](examples/test_geir_sgd.cpp) | Builds the graph with the [SGD operator IR](op_graph/sgd_proto.h) and calls the operator. |
