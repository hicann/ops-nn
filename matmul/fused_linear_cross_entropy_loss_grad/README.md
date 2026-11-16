# FusedLinearCrossEntropyLossGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                   |    √     |
| <term>Atlas A3系列产品</term>       |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                   |    ×     |
| <term>Atlas推理系列产品</term> |    ×     |
| <term>Atlas训练系列产品</term> |    ×     |

## 功能说明

- 算子功能：本算子是词汇表并行场景下交叉熵损失计算模块中的一部分，解决超大规模词汇表下的显存和计算效率问题，当前部分为梯度计算实现，用于计算叶子节点`input`和`weight`的梯度。
  需要获得`aclnnFusedLinearOnlineMaxSum`、`aclnnFusedCrossEntropyLossWithMaxSum`的相关输出，以及`logits`相关的全局通信结果作为本接口输入。
- 计算公式：

&emsp;&emsp;高性能模式，softmaxOptional非nullptr：

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

&emsp;&emsp;省显存模式，softmaxOptional为nullptr：

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

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| grad | 输入 | 表示上游损失对logits的梯度，即公式中的grad。shape为[BT]，BT为batch内token总数。支持空tensor（BT=0时非空输出置零）。 | FLOAT32 | ND |
| input | 输入 | 表示语言模型隐藏层输出，即公式中的input。shape为[BT, H]，H为隐藏层维度。与weight数据类型须一致。支持空tensor。 | BFLOAT16、FLOAT16 | ND |
| weight | 输入 | 表示语言模型输出层（LM head）权重，即公式中的weight。shape为[V, H]，V为当前词汇表分片大小。与input数据类型须一致。支持空tensor。 | BFLOAT16、FLOAT16 | ND |
| target_mask | 输入 | 表示逐行掩码，即公式中的target_mask，标记该行目标是否属于当前词汇表分片：1表示不属于当前分片（跳过one-hot减1），0表示按masked_target中的局部索引参与计算。BOOL按元素表示，长度须等于BT；UINT8按bit打包，长度乘以8须不小于BT，第i行对应target_mask[i / 8]的第(i % 8)位，最低有效位（LSB）优先。 | BOOL、UINT8 | ND |
| masked_target | 输入 | 表示逐行目标在当前词汇表分片内的局部索引，即公式中的masked_target。长度须等于BT；未掩码行必须满足0 <= masked_target[i] < V，掩码行不使用目标索引。取值范围为INT64或INT32可表示的非负整数。 | INT64、INT32 | ND |
| logits_max | 可选输入 | 表示全局通信得到的logits最大值，即公式中的logits_max，用于省显存模式（softmax为nullptr）的数值稳定。shape为[BT]。省显存模式下必填；高性能模式不使用，传nullptr。默认值为nullptr。 | FLOAT32 | ND |
| sum_exp_logits | 可选输入 | 表示全局通信得到的指数和，即公式中的sum_exp_logits，用于省显存模式（softmax为nullptr）的归一化。shape为[BT]。省显存模式下必填；高性能模式不使用，传nullptr。默认值为nullptr。 | FLOAT32 | ND |
| softmax | 可选输入 | 表示前序aclnnFusedCrossEntropyLossWithMaxSum算子输出的概率矩阵，即公式中的softmax。shape为[BT, V]。传有效指针时走高性能模式（免重算前向，本算子将按公式就地更新该输入缓冲区，调用方不应复用其内容）；传nullptr时走省显存模式。默认值为nullptr。 | FLOAT32 | ND |
| input_grad_out | 输出 | 表示叶子节点input的梯度，即公式中的grad_input。shape为[BT, H]，与input一致；数据类型与input一致。 | BFLOAT16、FLOAT16 | ND |
| weight_grad_out | 输出 | 表示叶子节点weight的梯度，即公式中的grad_weight。shape为[V, H]，与weight一致；数据类型与weight一致；BT=0时输出置零（非未初始化内存）。 | BFLOAT16、FLOAT16 | ND |
| label_smoothing | 可选属性 | 表示标签平滑系数。当前版本仅支持取值0.0（参数保留用于对齐前向接口签名），非0取值行为未定义。默认值为0.0。 | FLOAT | - |

## 约束说明

- 记BT为grad长度、H为input第2维长度、V为weight第1维长度。input和inputGradOut的shape为[BT, H]，weight和weightGradOut的shape为[V, H]。
- targetMask支持BOOL和UINT8。BOOL按元素表示逐行掩码，长度须等于BT；UINT8按bit打包，长度乘以8须不小于BT，第i行对应targetMask[i / 8]的第(i % 8)位，最低有效位（LSB）优先。
- targetMask=1表示该行目标不属于当前词汇表分片，跳过one-hot减1；targetMask=0时按maskedTarget中的局部索引减1。掩码不跳过后续梯度缩放和矩阵乘，公式中的target_mask指解包后的逐行掩码。
- maskedTarget长度须等于BT，未掩码行必须满足`0 <= maskedTarget[i] < V`。掩码行不使用目标索引；内核对越界索引仅做防御性skip，不执行clamp，不将越界值视为正常目标。
- 非空计算时，两种模式均要求H <= 65535。仅高性能模式（softmaxOptional非nullptr）额外要求BT <= 65536、128 <= V <= 65280，且`2 * align_up(4 * V, 64) <= UBBytes - 32`。UBBytes为每核UB容量（字节），align_up为向上对齐至64字节，故V实际可用上限还受UB容量约束。省显存模式无上述高性能BT/V上下限及此UB公式约束。
- 上述维度及UB条件不是tiling成功的充分保证，仍受Matmul tiling和平台资源限制。
- 参数校验通过后，input或weight为空时仅将非空输出置零；两个输出均为空时不下发计算。workspaceSize须使用第一段接口实际返回值。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|--------|---------|------|
| aclnn API | [test_aclnn_fused_linear_cross_entropy_loss_grad](./examples/test_aclnn_fused_linear_cross_entropy_loss_grad.cpp) | 通过[aclnnFusedLinearCrossEntropyLossGrad](./docs/aclnnFusedLinearCrossEntropyLossGrad.md)接口调用FusedLinearCrossEntropyLossGrad算子。 |
| PyTorch API | - | 通过torch_npu的[npu_fused_linear_cross_entropy_loss_with_max_sum_backward](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_linear_cross_entropy_loss_with_max_sum_backward.md)接口调用FusedLinearCrossEntropyLossGrad算子。**需torch_npu 2.12.0及以上版本**。接口详细说明见链接文档。 |
