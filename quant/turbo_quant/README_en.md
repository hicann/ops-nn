# TurboQuant

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | × |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

TurboQuant normalizes the KV latent of MLA by token and quantizes it into 4bit codebook indices, and separately outputs the L2 norm `scale` of each token. The input `latent` must have completed the signed Hadamard rotation, but it must not be normalized in advance.

For the `i`-th token, the operator first computes:

$$
scale_i = norm_i = \sqrt{\sum_d latent_{i,d}^2 + 10^{-16}}
$$

The normalized elements are:

$$
u_{i,d} = latent_{i,d} / scale_i
$$

The indices are selected according to the midpoints between adjacent codebook center points:

$$
nibble_{i,d} = \sum_{b=0}^{14} [u_{i,d} \ge (centroids_b + centroids_{b+1}) / 2]
$$

The indices of two adjacent dimensions are packed into one byte, where the low nibble stores the even-dimension index and the high nibble stores the odd-dimension index:

$$
y_{i,k} = nibble_{i,2k} \mathbin{|} (nibble_{i,2k+1} \ll 4)
$$

## Parameters

| Parameter name | Input/Output | Description | Data type | Data format | Dimensions (shape) |
| --- | --- | --- | --- | --- | --- |
| latent | Input | The unnormalized KV latent that has completed the signed Hadamard rotation | FLOAT32 | ND | `[numTokens, 512]` |
| centroids | Input | The 16 Lloyd-Max codebook center points sorted in ascending order | FLOAT32 | ND | `[16]` |
| y | Output | The quantization result after the two 4bit indices are packed, with the low nibble first | UINT8 | ND | `[numTokens, 256]` |
| scale | Output | The L2 norm of each token | FLOAT16 | ND | `[numTokens]` |

## Constraints

- The operator provides deterministic computation by default.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn API | [test_aclnn_turbo_quant](./examples/test_aclnn_turbo_quant.cpp) | Invoke the TurboQuant operator through the [aclnnTurboQuant](./docs/aclnnTurboQuant.md) interface. |
| PyTorch API | - | Invoke the TurboQuant operator through the [turbo_quant](./docs/torchapi_turbo_quant.md) interface. |
