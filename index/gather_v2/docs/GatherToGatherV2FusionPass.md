# GatherToGatherV2FusionPass

## 融合模式

该融合将符合图融合pattern的Gather算子改为GatherV2算子。

![](../../../docs/zh/figures/GatherToGatherV2FusionPass_1.png)

## 使用约束

该融合规则不能关闭。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->
