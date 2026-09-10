# SoftmaxGradExtV2FusionPass

## 融合模式

该融合将符合下图左侧图结构的Mul、ReduceSum、Sub这些小算子，融合成下图右侧的SoftmaxGradExt算子。

**场景一**
![](../../../docs/zh/figures/SoftmaxGradExtV2FusionPass_1.png)

**场景二**
![](../../../docs/zh/figures/SoftmaxGradExtV2FusionPass_2.png)

**场景三**
![](../../../docs/zh/figures/SoftmaxGradExtV2FusionPass_3.png)

**场景四**
![](../../../docs/zh/figures/SoftmaxGradExtV2FusionPass_4.png)

## 使用约束

- 输入约束：
  - Mul\_1节点的输入与Sub节点的第一个输入共用input0。
  - Mul\_1节点的输入与Mul\_2节点的输入共用input1。
  - ReduceSum的axis参数必须为-1或最后一维。

- 数据格式和shape约束：
  - input0、input1的shape必须是1D\~6D。
  - input0和input1的数据格式为ND。
  - input2可以是scalar，也可以是与input0相同shape的张量。

- 不支持动态shape场景。
- 数据类型约束：
  - input0、input1、input2的数据类型需要保持一致。
  - 数据类型支持FLOAT16、FLOAT32、BFLOAT16。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->
