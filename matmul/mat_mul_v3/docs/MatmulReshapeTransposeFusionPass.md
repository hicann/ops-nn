# MatmulReshapeTransposeFusionPass

## 融合模式

该融合将符合图融合pattern的Matmul算子前后的Transpose算子删除，同时交换Matmul算子的两路输入。

虚线框中的结构可以是一个或者多个。

![](../../../docs/zh/figures/MatmulReshapeTransposeFusionPass_1.png)

## 使用约束

- 算子dtype只支持FP32。
- format只支持ND/NCHW/NHWC。
- Matmul算子左矩阵输入必须为2维，右矩阵输入必须为三维，且右矩阵最后一维shape为1。
- 所有Transpose算子输入必须是3维，且最后一维shape为1，其功能为交换shape前两维，不支持动态shape。
- Matmul算子之前的Reshape算子，进行的操作是删除shape中最后一个1的维度。Matmul算子之后的Reshape算子则是在shape最后增加一个1的维度。
- Matmul算子的属性transpose\_x1/transpose\_x2必须都是false。
- 第一个Reshape算子之后，至少有一路Matmul+Reshape+Transpose的结构。支持多路Matmul+Reshape+Transpose，要求每一路都满足上述结构，否则不融合。

## 支持的型号

<!-- npu="310b" id1 -->
Atlas 200I/500 A2 推理产品
<!-- end id1 -->

<!-- npu="910b" id2 -->
Atlas A2 训练系列产品/Atlas A2 推理系列产品
<!-- end id2 -->

<!-- npu="A3" id3 -->
Atlas A3 训练系列产品/Atlas A3 推理系列产品
<!-- end id3 -->

<!-- npu="950" id4 -->
Ascend 950PR/Ascend 950DT
<!-- end id4 -->
