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

- Operator function: Logically concatenate hidden and optional markov_emb along the last dimension, project each row with weight, add optional bias, apply temperature scaling, optional clamp_low and clamp_high bounds, and then Sigmoid. Accumulation, post-processing, and output use FLOAT32.
- Formula:

  The following formulas use a 3D input. For a 2D input, omit the batch index $b$. $K_1$ is the last dimension of hidden. If markov_emb is absent, $K_2=0$; otherwise, $K_2$ is its last dimension. The length of weight is $K_1+K_2$.

  1. Project the logically concatenated features and add the bias:

     $$
     z_{b,n}=\sum_{k=0}^{K_1-1}\mathrm{hidden}_{b,n,k}\,\mathrm{weight}_k
     +\sum_{k=0}^{K_2-1}\mathrm{markov\_emb}_{b,n,k}\,\mathrm{weight}_{K_1+k}+p_n
     $$

     The second sum is $0$ when $K_2=0$. If bias is absent, $p_n=0$; otherwise, $p_n=\mathrm{bias}_n$ and is shared across batches.

  2. Apply temperature scaling to obtain the exponential input:

     $$
     T=\begin{cases}
     1, & \mathrm{temperature}=\mathrm{None}\text{ or }\mathrm{temperature}[0]=0,\\
     \mathrm{temperature}[0], & \text{otherwise},
     \end{cases}
     \qquad u_{b,n}=-\frac{z_{b,n}}{T}
     $$

  3. Apply clamp_low, then clamp_high, and compute Sigmoid. Let $C=\operatorname{FP32}(\mathrm{INT32\_MAX})=2147483648.0$:

     $$
     \begin{aligned}
     L&=\begin{cases}
     -C, & \mathrm{clamp\_low}=\mathrm{None},\\
     \operatorname{FP32}(\mathrm{clamp\_low}), & \text{otherwise},
     \end{cases}\\
     H&=\begin{cases}
     C, & \mathrm{clamp\_high}=\mathrm{None},\\
     \operatorname{FP32}(\mathrm{clamp\_high}), & \text{otherwise}
     \end{cases}
     \end{aligned}
     $$

     $$
     \begin{aligned}
     v_{b,n}&=\begin{cases}
     u_{b,n}, & L=-C,\\
     \max(u_{b,n},L), & L\ne-C,
     \end{cases}\\
     w_{b,n}&=\begin{cases}
     v_{b,n}, & H=C,\\
     \min(v_{b,n},H), & H\ne C,
     \end{cases}\\
     \mathrm{out}_{b,n}&=\frac{1}{1+\exp(w_{b,n})}
     \end{aligned}
     $$

     An explicitly supplied clamp_low equal to $-C$ or clamp_high equal to $C$ also skips the corresponding bound.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| hidden | Input | Features. The shape is [N,K1] or [B,N,K1]. | FLOAT16, BFLOAT16 | ND |
| weight | Input | Projection weight shared by all rows. The shape is [K1+K2], and the data type is the same as that of hidden. | FLOAT16, BFLOAT16 | ND |
| markov_emb | Optional input | The shape is [N,K2] or [B,N,K2]. The leading dimensions and data type are the same as those of hidden. When this parameter is absent, K2=0. | FLOAT16, BFLOAT16 | ND |
| temperature | Optional input | The shape is [1]. When this parameter is absent or the current value is positive or negative zero, the computation uses 1. | FLOAT32 | ND |
| bias | Optional input | The shape is [N]. The parameter is broadcast across batches. | FLOAT32 | ND |
| clamp_low | Optional attribute | Lower bound of the exponential input. The bound is skipped if omitted or if its FLOAT32 value equals -2147483648.0. | Float | - |
| clamp_high | Optional attribute | Upper bound of the exponential input. The bound is skipped if omitted or if its FLOAT32 value equals 2147483648.0. | Float | - |
| out | Output | The shape is [N] or [B,N]. | FLOAT32 | ND |

## Constraints

- Valid base formats and non-contiguous tensors are supported. FRACTAL_NZ is not supported.
- hidden, weight, and markov_emb must all be FLOAT16 or all be BFLOAT16. temperature, bias, and out are FLOAT32. The length of weight is $K_1+K_2$; the leading dimensions of markov_emb must match hidden; bias has shape [$N$]; and out has the shape of hidden without its last dimension.
- Each dimension, $B\times N$, and $K_1+K_2$ must be in $[0,\mathrm{UINT32\_MAX}]$. For a 2D input, $B=1$.
- When $K_1+K_2=0$, bias, temperature, optional clamping, and Sigmoid are still applied. When $B=0$ or $N=0$, the output is empty.

## Invocation Description

- <term>Ascend 950PR&950DT products</term>:

  | Invocation method | Sample code | Description |
  | --- | --- | --- |
  | aclnn API | [test_aclnn_confidence_head_fused](examples/arch35/test_aclnn_confidence_head_fused.cpp) | Invoke the ConfidenceHeadFused operator through<br>the [aclnnConfidenceHeadFused](docs/aclnnConfidenceHeadFused.md)<br>API. |
  | PyTorch API | [Invocation Example](docs/torchapi_confidence_head_fused.md#调用示例) | Invoke the ConfidenceHeadFused operator through<br>the [confidence_head_fused](docs/torchapi_confidence_head_fused.md)<br>API. |
