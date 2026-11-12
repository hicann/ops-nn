# L1LossGrad

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

- Operator function: Computes the gradient of the L1 loss (mean absolute error) with respect to the input predict.

- Formula:

  $$
  y = sign(predict - label) \times grads
  $$

  where sign(x) denotes the sign function of x:
  $$
  sign(x) =\begin{cases}
  1,& x>0 \\
  0,& x=0 \\
  -1,& x<0
  \end{cases}
  $$

  reduction is inherited from the forward loss and specifies the reduction mode of the loss: 'mean' divides each gradient element by the total number of elements; 'none' and 'sum' perform no extra scaling. In all three modes, the output y retains the broadcast shape of grads, predict and label.

## Parameters

| Parameter  | Input/Output/Attribute | Description                                                  | Data Type                | Format |
| :--------- | :--------------------- | :----------------------------------------------------------- | :----------------------- | :----- |
| grads      | Input                  | Backward gradient input, the grads in the formula.           | BFLOAT16, FLOAT16, FLOAT32 | ND     |
| predict    | Input                  | Predicted value, the predict in the formula.                 | BFLOAT16, FLOAT16, FLOAT32 | ND     |
| label      | Input                  | Ground-truth label, the label in the formula.                | BFLOAT16, FLOAT16, FLOAT32 | ND     |
| y          | Output                 | Computed output gradient, the y in the formula, whose shape is the broadcast result of grads, predict and label. | BFLOAT16, FLOAT16, FLOAT32 | ND     |
| reduction  | Optional attribute     | Specifies the reduction to be applied. The default value is "mean". The supported values are "none", "mean" and "sum". | STRING                   | -      |

## Constraints

- The data types of the inputs grads, predict and label must be the same as that of the output y.
- The shapes of the inputs grads, predict and label must satisfy the broadcast relationship, and the shape of the output y is the broadcast result of the three.
- Only the ND data format is supported.
- The output behavior is undefined when the input contains inf or nan. This operator provides no guarantee.

## Invocation Description

| Invocation method | Sample code                                                 | Description                                                  |
| :---------------- | :---------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn API         | [test_aclnn_l1_loss_grad.cpp](examples/test_aclnn_l1_loss_grad.cpp) | Invokes the L1LossGrad operator through the [aclnnL1LossBackward](docs/aclnnL1LossBackward.md) interface. |
| GE graph mode     | -                                                           | Invokes the L1LossGrad operator by constructing a graph through the [operator IR](op_graph/l1_loss_grad_proto.h) (REG_OP(L1LossGrad)). |
