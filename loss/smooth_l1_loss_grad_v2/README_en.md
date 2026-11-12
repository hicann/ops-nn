# SmoothL1LossGradV2

English | [简体中文](./README.md)

## Applicable Products

| Product                                                      | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>                               |    √     |
| <term>Atlas A2 products</term>                               |    √     |
| <term>Atlas 200I/500 A2 inference products</term>            |    ×     |
| <term>Atlas inference products</term>                        |    √     |
| <term>Atlas training products</term>                         |    √     |

## Function

- Operator function: Computes the gradient of the Smooth L1 loss with respect to the input predict.

- Formula:

  The backpropagation of the Smooth L1 loss can be computed by differentiation. For the case of |x-y|<sigma, the derivative is:
  $$
  \frac{\partial SmoothL1Loss(x,y)}{\partial x} = \frac{x-y}{sigma}
  $$

  For the case of |x-y|≥sigma, the derivative is:
  $$
  \frac{\partial SmoothL1Loss(x,y)}{\partial x} = sign(x - y)
  $$

  where x is predict and y is label; sigma is the threshold that switches between the L1 and L2 losses, and it must be non-negative. sign(x) denotes the sign function of x:
  $$
  sign(x) =\begin{cases}
  1,& x>0 \\
  0,& x=0 \\
  -1,& x<0
  \end{cases}
  $$

## Parameters

| Parameter  | Input/Output/Attribute | Description                                                  | Data Type                | Format |
| :--------- | :--------------------- | :----------------------------------------------------------- | :----------------------- | :----- |
| predict    | Input                  | The x in the formula, the predicted value.                   | BFLOAT16, FLOAT16, FLOAT32 | ND     |
| label      | Input                  | The y in the formula, the ground-truth label.                | BFLOAT16, FLOAT16, FLOAT32 | ND     |
| dout       | Input                  | Backward gradient input.                                     | BFLOAT16, FLOAT16, FLOAT32 | ND     |
| gradient   | Output                 | Computed output gradient, whose shape is the broadcast result of predict, label and dout. | BFLOAT16, FLOAT16, FLOAT32 | ND     |
| sigma      | Optional attribute     | The threshold that switches between the L1 and L2 losses. The default value is 1.0 and it must be non-negative. | FLOAT32                  | -      |
| reduction  | Optional attribute     | Specifies the reduction to be applied. The default value is "mean". The supported values are "none", "mean" and "sum". For the backward gradient: 'mean' divides each gradient element by the total number of elements; 'none' and 'sum' perform no extra scaling. | STRING                   | -      |

## Constraints

- The data types of the inputs predict, label and dout must be the same as that of the output gradient.
- The shapes of the inputs predict, label and dout must satisfy the broadcast relationship, and the shape of the output gradient is the broadcast result of the three.
- The rank upper limit of each input/output Tensor is 8; a rank-0 scalar input is normalized to {1}.
- Only the ND data format is supported.
- FLOAT16 and BFLOAT16 inputs are promoted to FLOAT32 inside the operator for computation and then written back as the output data type.
- When the input contains NaN/Inf, the output is propagated following the floating-point semantics.

## Invocation Description

| Invocation method | Sample code                                                 | Description                                                  |
| :---------------- | :---------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn API         | [test_aclnn_smooth_l1_loss_backward.cpp](examples/arch35/test_aclnn_smooth_l1_loss_backward.cpp) | Invokes the SmoothL1LossGradV2 operator through the [aclnnSmoothL1LossBackward](docs/aclnnSmoothL1LossBackward.md) interface. |
| GE graph mode     | -                                                           | Invokes the SmoothL1LossGradV2 operator by constructing a graph through the [operator IR](op_graph/smooth_l1_loss_grad_v2_proto.h) (REG_OP(SmoothL1LossGradV2)). |
