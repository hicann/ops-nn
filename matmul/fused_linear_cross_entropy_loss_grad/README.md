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

## 约束说明

- 记BT为grad长度、H为input第2维长度、V为weight第1维长度。input和inputGradOut的shape为[BT, H]，weight和weightGradOut的shape为[V, H]。
- targetMask支持BOOL和UINT8。BOOL按元素表示逐行掩码，长度须等于BT；UINT8按bit打包，长度乘以8须不小于BT，第i行对应targetMask[i / 8]的第(i % 8)位，最低有效位（LSB）优先。
- targetMask=1表示该行目标不属于当前词汇表分片，跳过one-hot减1；targetMask=0时按maskedTarget中的局部索引减1。掩码不跳过后续梯度缩放和矩阵乘，公式中的target_mask指解包后的逐行掩码。
- maskedTarget长度须等于BT，未掩码行必须满足`0 <= maskedTarget[i] < V`。掩码行不使用目标索引；内核对越界索引仅做防御性skip，不执行clamp，不将越界值视为正常目标。
- 非空计算时，两种模式均要求H <= 65535。仅高性能模式（softmaxOptional非nullptr）额外要求BT <= 65536、128 <= V <= 65280，且`2 * align_up(4 * V, 64) <= UBBytes - 32`。UBBytes为每核UB容量（字节），align_up为向上对齐至64字节，故V实际可用上限还受UB容量约束。省显存模式无上述高性能BT/V上下限及此UB公式约束。
- 上述维度及UB条件不是tiling成功的充分保证，仍受Matmul tiling和平台资源限制。
- 参数校验通过后，input或weight为空时仅将非空输出置零；两个输出均为空时不下发计算。workspaceSize须使用第一段接口实际返回值。

## 调用说明

| 调用方式   | 调用样例                              | 说明                                |
|-----------|-------------------------------------|-------------------------------------|
| aclnn调用 | [test_aclnn_fused_linear_cross_entropy_loss_grad](./examples/test_aclnn_fused_linear_cross_entropy_loss_grad.cpp) | 通过aclnnFusedLinearCrossEntropyLossGrad接口方式调用FusedLinearCrossEntropyLossGrad算子。 |
