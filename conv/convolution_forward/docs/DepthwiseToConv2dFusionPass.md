# DepthwiseToConv2dFusionPass

## 融合模式

该融合规则将DepthwiseConv2D算子转换为Conv2D算子。

![DepthwiseToConv2dFusionPass融合示意图](../../../docs/zh/figures/DepthwiseToConv2dFusionPass_1.png)

## 使用约束

- input的原始shape维度必须为4D。
- input的格式只支持NCHW或NHWC。
- input的通道数不能为-1（未知）。
<!-- npu="950,A3,910b" id4 -->
- 动态shape支持能力与型号相关：
  <!-- npu="950" id5 -->
  - Ascend 950PR&950DT系列产品：静态、动态shape场景均支持。
  <!-- end id5 -->
  <!-- npu="910b" id6 -->
  - Atlas A2系列产品：仅支持input为动态shape的场景，且input的通道维度必须为已知值。
  <!-- end id6 -->
  <!-- npu="A3" id7 -->
  - Atlas A3系列产品：仅支持input为动态shape的场景，且input的通道维度必须为已知值。
  <!-- end id7 -->
<!-- end id4 -->

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
