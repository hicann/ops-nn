# FusedLinearCrossEntropyLossGrad

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    √     |
| <term>Atlas A3 products</term>       |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                   |    ×     |
| <term>Atlas inference products</term> |    ×     |
| <term>Atlas training products</term> |    ×     |

## Function

- Description: This operator is part of the cross-entropy loss computation module in the vocabulary parallel scenario. It solves the memory and computation efficiency problems with an ultra-large vocabulary. The current part implements gradient computation, which is used to compute the gradients of the leaf nodes `input` and `weight`.
  It requires the related outputs of `aclnnFusedLinearOnlineMaxSum` and `aclnnFusedCrossEntropyLossWithMaxSum`, as well as the global communication result related to `logits`, as inputs of this interface.
- Formula:

&emsp;&emsp;High-performance mode, with softmaxOptional not nullptr:

$$
\text{softmax} \in \mathbb{R}^{BT \times V}
$$

$$
\text{arange\_1d} = [0, 1, \dots, BT-1] \in \mathbb{N}^{BT}
$$

$$
\text{softmax\_update} = \mathbf{1} - \text{target\_mask}.view(-1) \in \mathbb{R}^{BT}
$$

$$
\text{softmax}[\text{arange\_1d}, \text{masked\_target}] \leftarrow \text{softmax}[\text{arange\_1d}, \text{masked\_target}] - \text{softmax\_update}
$$

$$
\text{softmax} \leftarrow \text{softmax} \odot \text{grad}.unsqueeze(-1) \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_input} = \text{softmax} \cdot \text{weight} \in \mathbb{R}^{BT \times H}
$$

$$
\text{grad\_weight} = \text{softmax}^T \cdot \text{input} \in \mathbb{R}^{V \times H}
$$

&emsp;&emsp;Memory-saving mode, with softmaxOptional as nullptr:

$$
\text{vocab\_parallel\_logits} = \text{input} \cdot \text{weight}^T \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{logits\_sub} = \text{vocab\_parallel\_logits} - \text{logits\_max}.unsqueeze(-1) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{exp\_logits} = \exp(\text{logits\_sub}) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{exp\_logits} \gets \frac{\text{exp\_logits}}{\text{sum\_exp\_logits}.unsqueeze(-1)} \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_logits} = \text{exp\_logits} \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_2d} = \text{grad\_logits}.view(-1, \text{partition\_vocab\_size}) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{arange\_1d} = [0, 1, \dots, BT-1] \quad \in \mathbb{N}^{BT}
$$

$$
\text{softmax\_update} = 1 - \text{target\_mask}.view(-1) \quad \in \mathbb{R}^{BT}
$$

$$
\text{grad\_2d}[\text{arange\_1d}, \text{masked\_target\_1d}] \gets \text{grad\_2d}[\text{arange\_1d}, \text{masked\_target\_1d}] - \text{softmax\_update}
$$

$$
\text{grad\_logits} \gets \text{grad\_logits} \odot \text{grad}.unsqueeze(-1) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_input} = \text{grad\_logits} \cdot \text{weight} \quad \in \mathbb{R}^{BT \times H}
$$

$$
\text{grad\_weight} = \text{grad\_logits}^T \cdot \text{input} \quad \in \mathbb{R}^{V \times H}
$$

## Constraints

- Let BT be the length of grad, H the second dimension of input, and V the first dimension of weight. The shapes of input and inputGradOut are [BT, H], and the shapes of weight and weightGradOut are [V, H].
- targetMask supports BOOL and UINT8. BOOL represents one mask value per row, and its length must equal BT. UINT8 is bit-packed, and its length multiplied by 8 must be at least BT. Row i uses bit (i % 8) of targetMask[i / 8], with the least significant bit (LSB) first.
- targetMask=1 means that the row's target is outside the current vocabulary shard and skips the one-hot subtraction of 1. When targetMask=0, 1 is subtracted at the local index in maskedTarget. Masking does not skip subsequent gradient scaling or matrix multiplication. In the formulas, target_mask denotes the unpacked per-row mask.
- The length of maskedTarget must equal BT. Unmasked rows must satisfy `0 <= maskedTarget[i] < V`. Masked rows do not use their target indices. The kernel only skips out-of-range indices defensively; it does not clamp them or treat them as valid targets.
- For non-empty computation, both modes require H <= 65535. Only high-performance mode (softmaxOptional is not nullptr) additionally requires BT <= 65536, 128 <= V <= 65280, and `2 * align_up(4 * V, 64) <= UBBytes - 32`. UBBytes is the per-core UB capacity in bytes, and align_up rounds up to a multiple of 64 bytes. Therefore, the effective upper bound of V also depends on UB capacity. Memory-saving mode is not subject to these high-performance BT/V bounds or this UB formula.
- These dimension and UB conditions do not guarantee successful tiling; Matmul tiling and platform resource limitations still apply.
- After parameter validation succeeds, if input or weight is empty, only non-empty outputs are zero-filled. No computation is scheduled when both outputs are empty. Use the actual workspaceSize returned by the first-phase API.

## Invocation Description

| Invocation method   | Invocation sample                              | Description                                |
|-----------|-------------------------------------|-------------------------------------|
| aclnn invocation | [test_aclnn_fused_linear_cross_entropy_loss_grad](./examples/test_aclnn_fused_linear_cross_entropy_loss_grad.cpp) | Invokes the FusedLinearCrossEntropyLossGrad operator through the aclnnFusedLinearCrossEntropyLossGrad interface. |
| PyTorch interface | - | Invokes the FusedLinearCrossEntropyLossGrad operator through the torch_npu interface npu_fused_linear_cross_entropy_loss_with_max_sum_backward. **torch_npu 2.12.0 or later is required**. |
