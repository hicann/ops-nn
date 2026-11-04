# SmoothL1Loss

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------- | :------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>   |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Description: Computes the smooth L1 loss (element-wise, with no reduction) between the prediction and the label.
- Formula:

  $$
  \text{loss}(z) = \begin{cases}
  \frac{0.5 \times z^2}{\sigma} & \text{if } |z| < \sigma \\
  |z| - 0.5 \times \sigma & \text{otherwise}
  \end{cases}
  $$

  where $z = \text{predict} - \text{label}$, and $\sigma$ is the piecewise threshold (default 1.0).

## Parameters

|Parameter name|Input/Output/Attribute|Description|Data type|Data format|
|-----|-----------|----|---------|------|
|predict|Input|Denotes the prediction value, that is, `predict` in the formula.|BFLOAT16, FLOAT16, FLOAT32|ND|
|label|Input|Denotes the ground-truth label value, that is, `label` in the formula. The data type is consistent with that of predict.|BFLOAT16, FLOAT16, FLOAT32|ND|
|loss|Output|Denotes the smooth L1 loss value output, that is, `loss` in the formula. The data type is the same as that of predict.|BFLOAT16, FLOAT16, FLOAT32|ND|
|sigma|Attribute|Denotes the piecewise threshold, which is a non-negative finite value, with the default value 1.0.|Float|-|

## Constraints

- The shapes and dtypes of predict and label must be consistent (the numbers of dimensions are equal and each dimension is equal; in the dynamic shape scenario, a -1 dimension acts as a wildcard).
- The numbers of dimensions (rank) of predict and label do not exceed 8; otherwise, the operator compilation reports an error.
- sigma is an optional attribute, with the default value 1.0. It must be a non-negative value; a negative value causes an error in operator compilation.
- Empty tensors are supported (any dimension is 0): when there are zero elements, the operator performs zero iterations and directly executes successfully without dispatching the actual computation.
- Dynamic shapes (including dynamic ranges) are supported; in scenarios where the shape contains -1, the value is inferred from the actual input.
- The data format supports only ND.
- Deterministic computation: the same input always produces the same output.

## Invocation Description

| Invocation method | Invocation sample | Description |
|---------|---------|------|
| GE graph mode | [test_geir_smooth_l1_loss.cpp](examples/test_geir_smooth_l1_loss.cpp) | Invokes the SmoothL1Loss operator in GE graph mode. For the operator prototype definition, see [smooth_l1_loss_proto.h](op_graph/smooth_l1_loss_proto.h). |
