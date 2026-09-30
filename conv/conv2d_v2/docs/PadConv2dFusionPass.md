# PadConv2dFusionPass

## 融合模式

该融合将Pad/PadV3+Conv2d算子融合成Conv2d算子。

正向融合场景仅用于：Pad/PadV3+Conv2d图场景。
<!-- npu="A3,910b,910,310p,310b" id6 -->
反向融合场景仅用于：训练网络中和正向场景对应的反向过程，遇到Pad+Conv2DBackpropFilterD，会融合成新的Conv2DBackpropFilterD，消除Pad。遇到Conv2DBackpropInputD+Slice，会融合成新的Conv2DBackpropInputD，消除Slice。该场景适用的产品如下：
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

![PadConv2dFusionPass融合模式一](../../../docs/zh/figures/PadConv2dFusionPass_1.png)

## 使用约束

- 不支持动态shape。
- 不支持Pad/PadV3算子连接多个Conv2d结构，融合前的结构的第一个节点仅与后一个节点连接，不会连接多个节点。
- Pad/PadV3节点的控制边不能连接到被融合的Cube节点（Conv2d/Conv2DBackpropFilterD）。
- 不支持paddings值<0。
- PadV3算子只支持mode为constant且constant\_values为0（dtype为fp32）的场景下进行融合。
- Pad/PadV3算子的paddings输入必须是Const节点，且数据类型只支持int32或int64。
- Pad/PadV3算子的x输入格式只支持NCHW或NHWC。
- Pad/PadV3算子N/C维度pad只支持为0，支持在Conv2d的H/W维度补pad，融合后的pad大小要在[0, 255]区间内。
<!-- npu="A3,910b,910,310p,310b" id13 -->
- 融合后的pad_top和pad_bottom均小于kernel_h。该约束适用于如下产品型号：
    <!-- npu="310p" id14 -->
    - Atlas推理系列产品
    <!-- end id14 -->
    <!-- npu="310b" id15 -->
    - Atlas 200I/500 A2推理产品
    <!-- end id15 -->
    <!-- npu="910" id16 -->
    - Atlas训练系列产品
    <!-- end id16 -->
    <!-- npu="910b" id17 -->
    - Atlas A2系列产品
    <!-- end id17 -->
    <!-- npu="A3" id18 -->
    - Atlas A3系列产品
    <!-- end id18 -->
<!-- end id13 -->

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
