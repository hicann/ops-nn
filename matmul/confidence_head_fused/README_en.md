# ConfidenceHeadFused

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

The operator logically concatenates hidden and the optional markov_emb along the last dimension, performs matrix-vector multiplication with the 1D weight, adds the optional bias, applies temperature scaling, optional clamping, and Sigmoid, and outputs the FP32 confidence.

```text
features = cat([hidden, markov_emb], dim=-1)  # hidden is used when markov_emb is absent
z = features @ weight + bias                # 0 is used when bias is absent
T_eff = 1 if temperature is None or temperature[0] == 0 else temperature[0]
u = -z / T_eff
if clamp_low is enabled:  u = max(u, clamp_low)
if clamp_high is enabled: u = min(u, clamp_high)
out = 1 / (1 + exp(u))
```

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| hidden | Input | Features. The shape is [N,K1] or [B,N,K1]. | FLOAT16, BFLOAT16 | ND |
| weight | Input | Projection weight shared by all rows. The shape is [K1+K2], and the data type is the same as that of hidden. | FLOAT16, BFLOAT16 | ND |
| markov_emb | Optional input | The shape is [N,K2] or [B,N,K2]. The leading dimensions and data type are the same as those of hidden. When this parameter is absent, K2=0. | FLOAT16, BFLOAT16 | ND |
| temperature | Optional input | The shape is [1]. When this parameter is absent or the current value is positive or negative zero, the computation uses 1. | FLOAT32 | ND |
| bias | Optional input | The shape is [N]. The parameter is broadcast across batches. | FLOAT32 | ND |
| clamp_low | Optional attribute | Lower bound of the exponential input. If this parameter is not passed, no lower-bound clamping is performed. | Float | - |
| clamp_high | Optional attribute | Upper bound of the exponential input. If this parameter is not passed, no upper-bound clamping is performed. | Float | - |
| out | Output | The shape is [N] or [B,N]. | FLOAT32 | ND |

## Constraints

- Valid base formats are supported, and FRACTAL_NZ is not supported. The ACLNN internally converts the inputs to contiguous and unifies them into ND. Valid non-contiguous inputs and output views are supported.
- hidden, weight, and markov_emb must be all FLOAT16 or all BFLOAT16. temperature, bias, and out are FLOAT32.
- Each dimension, B*N, and K1+K2 must be in [0,UINT32_MAX]. Non-empty GEMV is limited by the L1 capacity for fully loading the weights and the on-chip capacity required by tiling.
- The kernel reads the current value of temperature. Negative values are allowed, and the input tensor is not modified.
- Clamping applies to -z/T_eff. Lower-bound clamping is performed first, and then upper-bound clamping is performed. The lower bound -2147483648.0 and upper bound 2147483648.0 are FP32 convention values that indicate skipping the corresponding clamping. When 0 is explicitly passed in, clamping is performed as usual.
- When K1+K2=0, post-processing is still performed. When B=0 or N=0, the output is empty.

## Invocation Description

- <term>Ascend 950PR&950DT products</term>:

  | Invocation method | Sample code | Description |
  | --- | --- | --- |
  | aclnn API | [test_aclnn_confidence_head_fused](examples/arch35/test_aclnn_confidence_head_fused.cpp) | Invoke the ConfidenceHeadFused operator through<br>the [aclnnConfidenceHeadFused](docs/aclnnConfidenceHeadFused.md)<br>API. |
  | PyTorch API | [Invocation Example](docs/torchapi_confidence_head_fused.md#调用示例) | Invoke the ConfidenceHeadFused operator through<br>the [confidence_head_fused](docs/torchapi_confidence_head_fused.md)<br>API. |
