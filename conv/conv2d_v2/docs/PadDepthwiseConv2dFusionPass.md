# PadDepthwiseConv2dFusionPass

## 融合模式

该融合规则将PadD+DepthwiseConv2D算子融合为DepthwiseConv2D算子。

![PadDepthwiseConv2dFusionPass融合示意图](../../../docs/zh/figures/PadDepthwiseConv2dFusionPass_1.png)

## 使用约束

- Pad不支持动态shape输入和动态shape输出。
- 融合前DepthwiseConv2D需要有padding属性，且必须为VALID。
- Pad的输出中DepthwiseConv2D输出节点个数只能为1且Pad节点的所有输出节点都必须为DepthwiseConv2D。
- Pad的paddings输入必须是Const节点，且数据类型只支持int32或int64。
- Pad算子的x输入格式只支持NCHW或NHWC。
- Pad算子只支持在DepthwiseConv2D的H/W维度补pad，N/C维度pad必须为0。
- 融合后的pad大小要在[0, 255]区间内。
- Pad节点不能带控制边。
<!-- npu="A3,910b,910,310p,310b" id6 -->
- 融合后的pad_top和pad_bottom均小于kernel_h。该约束适用于如下产品型号：
    <!-- npu="310p" id8 -->
    - Atlas推理系列产品
    <!-- end id8 -->
    <!-- npu="310b" id9 -->
    - Atlas 200I/500 A2推理产品
    <!-- end id9 -->
    <!-- npu="910" id10 -->
    - Atlas训练系列产品
    <!-- end id10 -->
    <!-- npu="910b" id11 -->
    - Atlas A2系列产品
    <!-- end id11 -->
    <!-- npu="A3" id12 -->
    - Atlas A3系列产品
    <!-- end id12 -->
<!-- end id6 -->

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
