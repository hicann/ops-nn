# Conv2DSqueezeBiasaddFusionPass

## 融合模式

该融合将Conv2D+Squeeze+BiasAdd转换成Conv2D+BiasAdd+Squeeze的结构。

![Conv2DSqueezeBiasaddFusionPass融合示意图](../../../docs/zh/figures/Conv2DSqueezeBiasaddFusionPass_1.png)

## 使用约束

- Squeeze节点的输入节点必须为Conv2D，Squeeze节点的输出Atlas A3系列产品必须为BiasAdd。
- BiasAdd节点的bias输入维度必须为1。
- BiasAdd节点的bias输入如果来自Variable节点，则不融合。
- BiasAdd节点两路输入必须都是静态shape，否则不融合。

## 支持的型号

<!-- npu="310p" id1 -->
Atlas推理系列产品
<!-- end id1 -->

<!-- npu="310b" id2 -->
Atlas 200I/500 A2推理产品
<!-- end id2 -->

<!-- npu="910" id3 -->
Atlas训练系列产品
<!-- end id3 -->

<!-- npu="910b" id4 -->
Atlas A2系列产品
<!-- end id4 -->

<!-- npu="A3" id5 -->
Atlas A3系列产品
<!-- end id5 -->

<!-- npu="950" id7 -->
Ascend 950PR&950DT系列产品
<!-- end id7 -->
