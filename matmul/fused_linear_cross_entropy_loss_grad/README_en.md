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

## Parameter Description

| Parameter | Input/Output/Attribute | Description | Data Type | Format |
|-----------|------------------------|-------------|------------|--------|
| grad | Input | Upstream gradient of the loss w.r.t. logits, i.e. grad in the formula. Shape [BT], where BT is the total number of tokens in the batch. Empty tensor supported (non-empty outputs are zero-filled when BT=0). | FLOAT32 | ND |
| input | Input | Hidden states of the language model, i.e. input in the formula. Shape [BT, H], where H is the hidden size. Data type must match weight. Empty tensor supported. | BFLOAT16, FLOAT16 | ND |
| weight | Input | LM head weight, i.e. weight in the formula. Shape [V, H], where V is the current vocab shard size. Data type must match input. Empty tensor supported. | BFLOAT16, FLOAT16 | ND |
| target_mask | Input | Per-row mask, i.e. target_mask in the formula, indicating whether the row's target belongs to the current vocab shard: 1 means not in this shard (one-hot subtraction skipped); 0 means the local index in masked_target is used. BOOL is element-wise with length equal to BT; UINT8 is bit-packed where length * 8 must be no less than BT, row i maps to bit (i % 8) of target_mask[i / 8], LSB first. | BOOL, UINT8 | ND |
| masked_target | Input | Per-row local target index within the current vocab shard, i.e. masked_target in the formula. Length must equal BT; unmasked rows must satisfy 0 <= masked_target[i] < V; masked rows do not use the index. Value range: non-negative integers representable by INT64 or INT32. | INT64, INT32 | ND |
| logits_max | Optional input | Global max of logits obtained via collective communication, i.e. logits_max in the formula, for numerical stability in memory-saving mode (softmax is nullptr). Shape [BT]. Required in memory-saving mode; unused in high-performance mode, pass nullptr. Default: nullptr. | FLOAT32 | ND |
| sum_exp_logits | Optional input | Global sum of exponentials obtained via collective communication, i.e. sum_exp_logits in the formula, for normalization in memory-saving mode (softmax is nullptr). Shape [BT]. Required in memory-saving mode; unused in high-performance mode, pass nullptr. Default: nullptr. | FLOAT32 | ND |
| softmax | Optional input | Probability matrix output by the preceding aclnnFusedCrossEntropyLossWithMaxSum operator, i.e. softmax in the formula. Shape [BT, V]. A valid pointer selects the high-performance mode (forward recomputation skipped; this input buffer is updated in place per the formula and must not be reused by the caller); nullptr selects the memory-saving mode. Default: nullptr. | FLOAT32 | ND |
| input_grad_out | Output | Gradient w.r.t. the leaf input, i.e. grad_input in the formula. Shape [BT, H], same as input; data type same as input. | BFLOAT16, FLOAT16 | ND |
| weight_grad_out | Output | Gradient w.r.t. the leaf weight, i.e. grad_weight in the formula. Shape [V, H], same as weight; data type same as weight; zero-filled when BT=0 (not uninitialized memory). | BFLOAT16, FLOAT16 | ND |
| label_smoothing | Optional attribute | Label smoothing factor. Only 0.0 is supported in this version (parameter reserved to align with the forward interface signature); behavior of non-zero values is undefined. Default: 0.0. | FLOAT | - |

## Constraints

- Let BT be the length of grad, H the second dimension of input, and V the first dimension of weight. The shapes of input and inputGradOut are [BT, H], and the shapes of weight and weightGradOut are [V, H].
- targetMask supports BOOL and UINT8. BOOL represents one mask value per row, and its length must equal BT. UINT8 is bit-packed, and its length multiplied by 8 must be at least BT. Row i uses bit (i % 8) of targetMask[i / 8], with the least significant bit (LSB) first.
- targetMask=1 means that the row's target is outside the current vocabulary shard and skips the one-hot subtraction of 1. When targetMask=0, 1 is subtracted at the local index in maskedTarget. Masking does not skip subsequent gradient scaling or matrix multiplication. In the formulas, target_mask denotes the unpacked per-row mask.
- The length of maskedTarget must equal BT. Unmasked rows must satisfy `0 <= maskedTarget[i] < V`. Masked rows do not use their target indices. The kernel only skips out-of-range indices defensively; it does not clamp them or treat them as valid targets.
- For non-empty computation, both modes require H <= 65535. Only high-performance mode (softmaxOptional is not nullptr) additionally requires BT <= 65536, 128 <= V <= 65280, and `2 * align_up(4 * V, 64) <= UBBytes - 32`. UBBytes is the per-core UB capacity in bytes, and align_up rounds up to a multiple of 64 bytes. Therefore, the effective upper bound of V also depends on UB capacity. Memory-saving mode is not subject to these high-performance BT/V bounds or this UB formula.
- These dimension and UB conditions do not guarantee successful tiling; Matmul tiling and platform resource limitations still apply.
- After parameter validation succeeds, if input or weight is empty, only non-empty outputs are zero-filled. No computation is scheduled when both outputs are empty. Use the actual workspaceSize returned by the first-phase API.

## Invocation Description

| Invocation method | Sample code | Description |
|-------------------|-------------|-------------|
| aclnn API | [test_aclnn_fused_linear_cross_entropy_loss_grad](./examples/test_aclnn_fused_linear_cross_entropy_loss_grad.cpp) | Invokes the FusedLinearCrossEntropyLossGrad operator through the [aclnnFusedLinearCrossEntropyLossGrad](./docs/aclnnFusedLinearCrossEntropyLossGrad_en.md) interface. |
| PyTorch API | - | Invokes the FusedLinearCrossEntropyLossGrad operator through the torch_npu interface [npu_fused_linear_cross_entropy_loss_with_max_sum_backward](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_linear_cross_entropy_loss_with_max_sum_backward.md). **torch_npu 2.12.0 or later is required**. See the linked document for interface details. |
