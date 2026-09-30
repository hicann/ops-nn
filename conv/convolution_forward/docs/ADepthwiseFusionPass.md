# ADepthwiseFusionPass

## 融合模式

该融合规则将图中Depthwise算子的filter输入端插入Reshape算子，修改适配filter的shape，用于后续插入Transdata算子，确保格式转换正确。

### 融合场景一

![ADepthwiseFusionPass融合示意图1](../../../docs/zh/figures/ADepthwiseFusionPass_1.png)

### 融合场景二

![ADepthwiseFusionPass融合示意图2](../../../docs/zh/figures/ADepthwiseFusionPass_2.png)

### 融合场景三

![ADepthwiseFusionPass融合示意图3](../../../docs/zh/figures/ADepthwiseFusionPass_3.png)

## 使用约束

- DepthwiseConv2D的filter shape维度必须为4D。
- filter的格式只支持NCHW或HWCN。
- filter的Cin必须大于1（等于1时图结构不发生变化）。

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

<!-- npu="950" id6 -->
Ascend 950PR&950DT系列产品
<!-- end id6 -->
