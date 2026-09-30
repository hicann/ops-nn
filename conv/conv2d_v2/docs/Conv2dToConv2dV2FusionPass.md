# Conv2dToConv2dV2FusionPass

## 融合模式

该融合规则将Conv2d算子转换为Conv2dv2算子，融合后的Conv2dv2算子与原Conv2d算子输入、输出以及属性完全一致。

![Conv2dToConv2dV2FusionPass融合示意图](../../../docs/zh/figures/Conv2dToConv2dV2FusionPass_1.png)

## 使用约束

input的数据类型仅支持：Fp32、BF16、FP16、HIFloat8（HIFloat8场景bias为Fp32）。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->
