# ActsULQInputGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: ActsULQInputGrad is the backward input gradient operator of ActsULQ (Activations Universal Linear Quantization simulation), used in quantization-aware training (QAT) scenarios. In the forward stage, the lower-bound mask clamp_min_mask and the upper-bound mask clamp_max_mask are generated based on the clamp boundaries. Based on these masks, this operator elementwise gates the upstream gradient y_grad to obtain the input gradient x_grad.

- Formula:

  $$
  x\_grad = y\_grad \times clamp\_min\_mask \times clamp\_max\_mask
  $$

  That is, a gradient is passed through only when the element falls within both the quantization lower bound and upper bound (both masks are 1); otherwise it is set to zero.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
|--------|---------------|------|----------|----------|
| y_grad | Input | Denotes the gradient tensor passed in from upstream, corresponding to y_grad in the formula. Its shape must be exactly the same as clamp_min_mask, clamp_max_mask, and x_grad. | FLOAT, FLOAT16 | ND |
| clamp_min_mask | Input | Denotes the lower-bound clamp mask tensor, corresponding to clamp_min_mask in the formula, with values {0,1}. Its data type is bool or the same floating-point type as y_grad. Its shape must match y_grad. | BOOL, FLOAT16, FLOAT | ND |
| clamp_max_mask | Input | Denotes the upper-bound clamp mask tensor, corresponding to clamp_max_mask in the formula, with values {0,1}. Its data type must be the same as clamp_min_mask. Its shape must match y_grad. | BOOL, FLOAT16, FLOAT | ND |
| x_grad | Output | The gated input gradient tensor, corresponding to x_grad in the formula. Its data type must be the same as y_grad, and its shape must match y_grad. | FLOAT, FLOAT16 | ND |

## Constraints

- The data types of y_grad and x_grad must be the same, and only FLOAT and FLOAT16 are supported.
- The data types of clamp_min_mask and clamp_max_mask must be the same. Floating-point masks must have the same dtype as y_grad, or be bool. Only 4 valid dtype combinations exist:

  | y_grad | clamp_min_mask / clamp_max_mask | x_grad |
  |--------|---------------------------------|--------|
  | FLOAT16 | BOOL | FLOAT16 |
  | FLOAT16 | FLOAT16 | FLOAT16 |
  | FLOAT | BOOL | FLOAT |
  | FLOAT | FLOAT | FLOAT |

- The semantic values of clamp_min_mask and clamp_max_mask should be 0 or 1.
- The shapes of the three inputs and the output must be exactly the same (no broadcasting), and empty tensors are supported.

## Invocation Description

| Invocation method | Sample code | Description |
|----------|----------|------|
| Graph mode | [test_geir_acts_ulq_input_grad](./examples/test_geir_acts_ulq_input_grad.cpp) | Calls the ActsULQInputGrad operator by building the graph through the [operator IR](./op_graph/acts_ulq_input_grad_proto.h). |
