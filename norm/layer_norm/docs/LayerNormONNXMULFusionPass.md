# LayerNormONNXMULFusionPass

## 融合模式

该融合规则将ONNX导入的ReduceMean、两个独立的Sub、Mul（或Pow）、Add、Sqrt（或Pow）、Div等小算子组合识别并融合为LayerNormV3算子。带缩放和偏移的场景同时融合末尾的Mul和Add。

- 场景一：使用Mul计算平方，带gamma和beta。

  ![](../../../docs/zh/figures/layer_norm_onnx_mul_fusion_pass_1.png)

  **融合为**

  ![](../../../docs/zh/figures/layer_norm_onnx_mul_fusion_pass_3.png)

- 场景二：使用Mul计算平方，不带gamma和beta。

  ![](../../../docs/zh/figures/layer_norm_onnx_mul_fusion_pass_2.png)

  **融合为**

  ![](../../../docs/zh/figures/layer_norm_onnx_mul_fusion_pass_4.png)

- 场景三：使用Pow计算平方，带gamma和beta。

  ![](../../../docs/zh/figures/layer_norm_onnx_mul_fusion_pass_5.png)

  **融合为**

  ![](../../../docs/zh/figures/layer_norm_onnx_mul_fusion_pass_3.png)

- 场景四：使用Pow计算平方，不带gamma和beta。

  ![](../../../docs/zh/figures/layer_norm_onnx_mul_fusion_pass_6.png)

  **融合为**

  ![](../../../docs/zh/figures/layer_norm_onnx_mul_fusion_pass_4.png)

场景一、二中，Mul\_0的两个输入均为Sub\_1的结果，用于计算平方；场景三、四中，Pow以Sub\_1的结果为底数，指数输入exp为2。上述各场景的Sqrt均支持替换为exp为0.5的Pow。Add\_0的两个输入可以交换，带gamma和beta时，Mul\_1、Add\_1的两个输入也可以分别交换。

融合后，LayerNormV3的输入为x、gamma和beta；带gamma和beta时复用原输入，不带时创建全1的gamma和全0的beta常量，shape均为[x最后一维长度]。epsilon常量转为LayerNormV3的epsilon属性，begin\_norm\_axis保留ReduceMean\_0的原始轴值，begin\_params\_axis固定为-1。只使用LayerNormV3的y输出替换原子图出口，不额外连接mean和rstd输出。

## 使用约束

- x的维数必须大于等于1，只支持沿最后一维归一化。
- 两个ReduceMean的axes必须是INT32或INT64常量，均只包含一个元素，且原始轴值必须相同；轴值为-1或x维数减1。两个ReduceMean的keep\_dims均必须为true。
- Sub\_0和Sub\_1必须是两个独立的节点，不能复用同一个Sub。两个Sub的第一个输入均为x，第二个输入均为ReduceMean\_0的结果，不能调换位置。Sub\_0用于Div的分子，Sub\_1用于平方计算。
- epsilon必须是FLOAT32或FLOAT16常量，shape为标量或[1]。
- 使用Pow时，指数输入exp必须是FLOAT32或FLOAT16的标量或[1]常量；平方支路的exp为2，开方支路的exp为0.5。
- 带gamma和beta时，二者必须是一维张量，且长度相同，可以为常量或非常量。
- 不带gamma和beta时，x的最后一维必须是已知的正数，Div输出的数据类型仅支持FLOAT32、FLOAT16，用于创建gamma和beta常量。其他维度允许为动态维度。
- 被融合子图的内部节点不能被子图外的其他分支复用；子图出口可以有多个消费者。不带gamma和beta的模式以Div为出口，带gamma和beta的模式以Add\_1为出口。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->
