# InternVlAddRmsNorm

## Product Support

| Product | Support |
|:--|:--:|
| Atlas A3 training/inference series | Yes |
| Atlas A2 training/inference series | Yes |

## Function

`InternVlAddRmsNorm` computes fused residual add and RMSNorm:

    residual_out = x + residual
    y = residual_out / sqrt(mean(residual_out * residual_out) + epsilon) * gamma

For an input `x` with shape `[..., hidden_size]`, `residual` has the same shape as `x`,
`gamma` has shape `[hidden_size]`, and both outputs `y` and `residual_out` have the
same shape as `x`.

## Parameters

| Name | Input/Output/Attr | Description | Data Type | Format |
|:--|:--|:--|:--|:--|
| x | Input | Input tensor. The last dimension is normalized. | FLOAT16, FLOAT, BFLOAT16 | ND |
| residual | Input | Residual tensor added to `x`. It must have the same shape as `x`. | FLOAT16, FLOAT, BFLOAT16 | ND |
| gamma | Input | RMSNorm scale tensor. Its shape must match the last dimension of `x`. | FLOAT16, FLOAT, BFLOAT16 | ND |
| y | Output | Normalized output tensor. It has the same shape as `x`. | FLOAT16, FLOAT, BFLOAT16 | ND |
| residual_out | Output | Result of `x + residual`. It has the same shape as `x`. | FLOAT16, FLOAT, BFLOAT16 | ND |
| epsilon | Attr | Small value added to the variance term for numerical stability. Default is `1e-6`. | float | - |

## Constraints

- `x` rank must be greater than or equal to 2.
- `x`, `residual`, `y`, and `residual_out` must have the same shape.
- `gamma` must match the last dimension of `x`.
- All inputs and outputs must use the same data type.
- Supported data types are `FLOAT16`, `FLOAT`, and `BFLOAT16`.

## Invocation

| Invocation | Example | Description |
|:--|:--|:--|
| ACLNN | - | Calls the operator through [aclnnInternVlAddRmsNorm](./docs/aclnnInternVlAddRmsNorm.md). |
| Graph mode | - | Builds the operator through [operator IR](./op_graph/intern_vl_add_rms_norm_proto.h). |

## Contribution

- Contributor: China Unicom (Guangdong) Industrial Internet Co., Ltd.
