# GroupNorm

English | [简体中文](./README.md)

## Applicable Products

<!-- markdownlint-disable MD033 -->

| Product | Support |
| :----------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

<!-- markdownlint-enable MD033 -->

## Function

- Operator function: GroupNorm divides the channel axis of input `x` into `num_groups` groups and independently computes the overall mean and overall variance for each sample and each group.

- Formula:

  $$
  y = \frac{x - mean}{\sqrt{variance + eps}} \times gamma + beta
  $$

  The shapes of the outputs `mean` and `variance` are both `(N, num_groups)`. `data_format` and `is_training` are used only for compatibility with the existing GEIR interface and do not participate in Kernel computation.

## Parameters

The following parameter descriptions apply to GE graph mode. For the aclnn API parameter descriptions, refer to [aclnnGroupNorm.md](docs/aclnnGroupNorm_en.md).

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Input `x` in the formula. Its shape has at least 2 dimensions and is interpreted as `(N, C, *)`. | FLOAT16, FLOAT | ND |
| gamma | Input | `gamma` in the formula, a one-dimensional tensor of length `C`, with the same type as `x`. | FLOAT16, FLOAT | ND |
| beta | Input | `beta` in the formula, a one-dimensional tensor of length `C`, with the same type as `x`. | FLOAT16, FLOAT | ND |
| num_groups | Required attribute | Number of channel groups. It is a positive integer, and `C` must be divisible by it. | INT | - |
| data_format | Optional attribute | Input data format. The default value is `NCHW`. Currently it is used only for interface compatibility. | STRING | - |
| eps | Optional attribute | Numerical stability term added to the variance. The default value is `0.0001`. | FLOAT | - |
| is_training | Optional attribute | Whether it is training mode. The default value is `true`. Currently it is used only for interface compatibility. | BOOL | - |
| y | Output | Output `y` in the formula. Its shape and data type are the same as those of `x`. | FLOAT16, FLOAT | ND |
| mean | Output | Mean of each group, with shape `(N, num_groups)`. | FLOAT16, FLOAT | ND |
| variance | Output | Overall variance of each group, with shape `(N, num_groups)`. | FLOAT16, FLOAT | ND |

## Constraints

- In GE graph mode, the data types of all inputs and outputs must be the same.
- Ascend 950PR&950DT products:
  - GE graph mode requires `C` to be greater than 0. When `N=0`, all three outputs are empty and no Kernel is dispatched.
  - In GE graph mode, empty tensors containing zero dimensions are not supported when `N` is greater than 0, and no Kernel is dispatched after Host validation fails.

## Invocation Description

| Invocation method | Invocation sample | Description |
| :--- | :--- | :--- |
| GE graph mode | [test_geir_group_norm.cpp](examples/test_geir_group_norm.cpp) | Invokes the operator through [GroupNorm IR](op_graph/group_norm_proto.h) graph construction. |
| aclnn API | [test_aclnn_GroupNorm.cpp](examples/test_aclnn_GroupNorm.cpp) | Invokes the operator through the [aclnnGroupNorm](docs/aclnnGroupNorm_en.md) interface. |
