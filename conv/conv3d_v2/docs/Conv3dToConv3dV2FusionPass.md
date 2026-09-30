# Conv3dToConv3dV2FusionPass

## 融合模式

该融合规则将Conv3d算子转换为Conv3dv2算子，融合后的Conv3dv2算子与原Conv3d算子输入、输出以及属性完全一致。

![Conv3dToConv3dV2FusionPass融合示意图](../../../docs/zh/figures/Conv3dToConv3dV2FusionPass_1.png)

## 使用约束

- 不支持Conv3d输出节点为FixPipe算子的场景。
- 当input数据类型为FP16时，额外约束如下：
  - input的输入节点为TransData且该TransData为NDHWC转NDC1HWC0（C维度为16的整数倍）的场景不融合，其余TransData场景不影响融合。
  - Conv3d输出节点的任意一路输入节点为IFMR算子时不融合。
<!-- npu="950" id8 -->
- 仅支持动态shape场景，支持的数据类型为Fp32/BF16/FP16/HIFloat8（HIFloat8场景bias为Fp32）。该约束适用于如下产品型号：
  <!-- npu="950" id4 -->
  - Ascend 950PR&950DT系列产品
  <!-- end id4 -->
<!-- end id8 -->
<!-- npu="A3,910b" id5 -->
- 仅支持静态shape场景，支持的数据类型为Fp32/BF16/FP16（BF16场景bias为Fp32），融合后的Conv3dV2不支持Hf32。该约束适用于如下产品型号：
  <!-- npu="910b" id6 -->
  - Atlas A2系列产品
  <!-- end id6 -->
  <!-- npu="A3" id7 -->
  - Atlas A3系列产品
  <!-- end id7 -->
<!-- end id5 -->

## 支持的型号

<!-- npu="910b" id2 -->
Atlas A2系列产品
<!-- end id2 -->

<!-- npu="A3" id3 -->
Atlas A3系列产品
<!-- end id3 -->

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->
