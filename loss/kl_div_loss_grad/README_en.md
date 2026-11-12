# KlDivLossGrad

## Product Support

| Product                                                      | Supported |
| :----------------------------------------------------------- | :-------: |
| <term>Ascend 950PR&950DT products</term>                      |     √     |
| <term>Atlas A3 products</term>                                |     √     |
| <term>Atlas A2 products</term>                                |     √     |
| <term>Atlas 200I/500 A2 inference products</term>                       |     ×     |
| <term>Atlas inference products</term>                               |     ×     |
| <term>Atlas training products</term>                               |     ×     |

## Function Description

- Operator function: Computes the gradient of the KL divergence loss with respect to the input `input`.

- Computation formula:

  - When `log_target` is `false`:

    $$
    y = -target \times grad
    $$

  - When `log_target` is `true`:

    $$
    y = -e^{target} \times grad
    $$

  Where:

  - `grad` is the gradient input of the backward propagation.
  - `target` is the ground-truth label.
  - `y` is the computed output gradient.
  - `log_target` indicates whether `target` is already in log space.

- `reduction` scaling:

  - `'none'`: no scaling.
  - `'sum'`: no scaling.
  - `'mean'`: the output is divided by the total number of elements in the output.
  - `'batchmean'`: the output is divided by the number of batches, that is, the size of the first dimension of `input`.

## Parameter Description

| Parameter  | Input/Output/Attribute | Description                                                  | Data Type                | Format |
| :--------- | :--------------------- | :----------------------------------------------------------- | :----------------------- | :----- |
| grad       | Input                  | Gradient input of the backward propagation, corresponding to `grad` in the formula. | BFLOAT16, FLOAT16, FLOAT32   | ND     |
| input      | Input                  | Input tensor, the target for which the gradient is computed, corresponding to `input` in the formula. | BFLOAT16, FLOAT16, FLOAT32   | ND     |
| target     | Input                  | Ground-truth label, corresponding to `target` in the formula. | BFLOAT16, FLOAT16, FLOAT32   | ND     |
| y          | Output                 | Computed output gradient, corresponding to `y` in the formula. Its shape is the same as that of `input`. | BFLOAT16, FLOAT16, FLOAT32   | ND     |
| reduction  | Optional attribute     | Specifies the reduction to be applied to the output. The default value is "mean". Supports "none", "mean", "sum", and "batchmean". | STRING                   | -      |
| log_target | Optional attribute     | Specifies whether to convert `target` to log space. The default value is false. | BOOL                     | -      |

## Constraints

- The shapes of `input` and `target` must satisfy the broadcast relationship, and the shape of `grad` must be able to broadcast to the shape of `input` and `target` after broadcasting.
- The data types of `grad`, `input`, `target`, and `y` must be the same.
- When `reduction` is "mean", `input` cannot be an empty Tensor, that is, the length of each dimension of `input` cannot be 0.
- When `reduction` is "batchmean", the length of the first dimension of `input` must be greater than 0.
- When `log_target` is `true`, the computation formula contains `exp(target)`. If the value of `target` is too large, `exp(target)` overflows and the output may be inf or nan.

## Usage

| Calling Method | Sample Code                                                 | Description                                                  |
| :------------- | :---------------------------------------------------------- | :----------------------------------------------------------- |
| aclnn API      | [test_aclnn_kl_div_loss_grad.cpp](examples/test_aclnn_kl_div_loss_grad.cpp) | Calls the KlDivLossGrad operator through the [aclnnKlDivBackward](docs/aclnnKlDivBackward.md) interface. |
| GE graph mode  | -                                                           | Calls the KlDivLossGrad operator by composing the graph through the [operator IR](op_graph/kl_div_loss_grad_proto.h). |
