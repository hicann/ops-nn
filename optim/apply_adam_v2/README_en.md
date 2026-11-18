# ApplyAdamV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:-----|:-------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: implements parameter updates of the Adam (Adaptive Moment Estimation) optimizer. It supports the `adam` and `mbart_adam` modes with optional gradient clipping, and is used after backpropagation in deep learning training.
- Formula:

  **Standard Adam mode** (`adam_mode = "adam"`):

  $$
  g_t = \text{grad} \cdot \text{clip\_coeff}
  $$

  $$
  m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
  $$

  $$
  v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
  $$

  $$
  u = \frac{m_t}{\sqrt{v_t} + \epsilon}
  $$

  $$
  \text{var}_t = \text{var}_{t-1} - \eta \cdot (u + \text{weight\_decay} \cdot \text{var}_{t-1})
  $$

  **mBART Adam mode** (`adam_mode = "mbart_adam"`):

  $$
  m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot \text{grad}
  $$

  $$
  v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot \text{grad}^2
  $$

  $$
  u = \frac{m_t}{\sqrt{v_t} + \epsilon}
  $$

  $$
  \text{var}_t = \text{var}_{t-1} - \left(\text{step\_size} \cdot u + \text{weight\_decay} \cdot (\eta \cdot \text{var}_{t-1})\right)
  $$

  Here $\mathrm{var}$ is the model parameter, $m$ is the first moment estimate, $v$ is the second moment estimate, $\beta_1$ and $\beta_2$ are decay rates, $\eta$ is the learning rate, and $\epsilon$ is the numerical stability constant. $\mathrm{clip\_coeff}$ is the gradient scaling coefficient; its value is described in [Constraints](#constraints). $\mathrm{weight\_decay}$ is the weight decay coefficient, and $\mathrm{step\_size}$ is the update step size.

## Parameters

Tensor denotes a multidimensional array, shape denotes its dimensions, and dtype denotes its data type. ND is the data format for multidimensional tensors; see [Data Format](../../docs/en/context/data_format.md).

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
|:-------|:--------------|:-----|:--------|:--------|
| var | Input | Model parameters to be updated, corresponding to $\mathrm{var}_{t-1}$ in the formula. | FLOAT, FLOAT16 | ND |
| m | Input | First moment estimate, corresponding to $m_{t-1}$ in the formula. The shape is consistent with var. | FLOAT, FLOAT16 | ND |
| v | Input | Second moment estimate, corresponding to $v_{t-1}$ in the formula. The shape is consistent with var. | FLOAT, FLOAT16 | ND |
| lr | Input | Learning rate, corresponding to $\eta$ in the formula, a scalar. | FLOAT, FLOAT16 | ND |
| beta1 | Input | First moment decay rate, corresponding to $\beta_1$ in the formula, a scalar, typically 0.9. | FLOAT, FLOAT16 | ND |
| beta2 | Input | Second moment decay rate, corresponding to $\beta_2$ in the formula, a scalar, typically 0.999. | FLOAT, FLOAT16 | ND |
| epsilon | Input | Numerical stability constant, corresponding to $\epsilon$ in the formula, a scalar, typically `1e-8`. | FLOAT, FLOAT16 | ND |
| grad | Input | Gradient tensor. The shape is consistent with var. | FLOAT, FLOAT16 | ND |
| max_grad_norm | Input (optional) | Scalar that controls gradient clipping, enabled when the value is greater than 0. Used only in `adam` mode. | FLOAT, FLOAT16 | ND |
| global_grad_norm | Input | Global gradient norm, a scalar. Used only in `adam` mode. | FLOAT, FLOAT16 | ND |
| weight_decay | Input | Weight decay coefficient, a scalar, typically 0.01. | FLOAT, FLOAT16 | ND |
| step_size | Input (optional) | Update step size, a scalar. Used only in `mbart_adam` mode. | FLOAT, FLOAT16 | ND |
| adam_mode | Attribute | Computation mode. `"adam"` selects standard Adam mode, and `"mbart_adam"` selects mBART Adam mode. The default is `"adam"`. | STRING | - |
| var | Output | Updated model parameters, corresponding to $\mathrm{var}_t$ in the formula. The shape and dtype are consistent with the input var. | FLOAT, FLOAT16 | ND |
| m | Output | Updated first moment estimate, corresponding to $m_t$ in the formula. The shape and dtype are consistent with the input m. | FLOAT, FLOAT16 | ND |
| v | Output | Updated second moment estimate, corresponding to $v_t$ in the formula. The shape and dtype are consistent with the input v. | FLOAT, FLOAT16 | ND |

## Constraints

- The shapes and data types of the four tensors var, m, v, and grad must be exactly the same.
- lr, beta1, beta2, epsilon, max_grad_norm, global_grad_norm, weight_decay, and step_size are scalar tensors with shape `[]` or `[1]`. Their data types must be the same as the data type of var.
- For FLOAT16 (16-bit floating-point) input, the internal computation uses FLOAT32 (32-bit floating-point), and the output is converted back to FLOAT16.
- In `adam` mode, gradient clipping is enabled only when max_grad_norm is provided and its value is greater than 0. When global_grad_norm is not a number (NaN), infinite (Inf), or less than `1e-10`, it is treated as 1. After this processing, if the value is greater than 1, `clip_coeff = 1 / global_grad_norm`; otherwise, `clip_coeff = 1`. When clipping is not enabled, `clip_coeff = 1`. max_grad_norm is a clipping switch; its value is not used as the scaling numerator.
- In `mbart_adam` mode, no gradient clipping is performed. If step_size is not provided, lr is used as step_size.
- The operator is deterministic by default, and the same input produces the same output.
- An empty tensor (a tensor with any dimension of length 0) with valid metadata returns three empty outputs with the same shape and dtype as the input, without numeric computation.
- The tensor rank ranges from 0 to 8.
- Dynamic shape support: Kernel mode supports `-1` (an unknown dimension) and `-2` (an unknown rank). Graph Engine (GE) graph mode does not support `-1` or `-2`, because the built-in shape inference in canndev cannot correctly propagate unknown dimensions to in-place outputs. This description applies to graph mode; for the shape constraints of the aclnn interface, see [aclnnApplyAdamV2](docs/aclnnApplyAdamV2.md#约束说明).

### Numerical Range

This operator supports inputs that conform to the Adam parameter update semantics. The caller must ensure the following numerical ranges:

| Parameter | Supported range |
|:-----|:---------|
| var, m, grad | Finite real numbers; positive values, negative values, and 0 are allowed, with no additional limit on the magnitude of the gradient or the first moment. |
| v | Finite and non-negative, denoting the second moment estimate of the squared gradient. |
| beta1, beta2 | The configured value satisfies `0 <= beta < 1`. For FLOAT16 input, the boundary case where the configuration is rounded to `1` during conversion is allowed. |
| lr | Finite and non-negative. |
| epsilon | Finite and non-negative; positive values are typically used. |
| weight_decay | Finite and non-negative. |
| step_size | Finite and non-negative in `mbart_adam` mode; in `adam` mode it does not participate in the computation, and its value is not required to satisfy this range. |

## Invocation Description

The `aclnnApplyAdamV2` interface provided by this repository supports Ascend 950PR&950DT products. This support scope applies to this interface. For its prototypes, in-place update semantics, and parameter constraints, see [aclnnApplyAdamV2](docs/aclnnApplyAdamV2.md).

| Invocation method | Invocation sample | Description |
|:--------|:--------|:-----|
| aclnn invocation | [test_aclnn_apply_adam_v2](examples/arch35/test_aclnn_apply_adam_v2.cpp) | Call `aclnnApplyAdamV2GetWorkspaceSize`, then `aclnnApplyAdamV2` to update var, m, and v in place. |
| Graph-mode invocation | [test_geir_apply_adam_v2](examples/test_geir_apply_adam_v2.cpp) | Calls the ApplyAdamV2 operator by building a graph through [operator IR](op_graph/apply_adam_v2_proto.h). |
