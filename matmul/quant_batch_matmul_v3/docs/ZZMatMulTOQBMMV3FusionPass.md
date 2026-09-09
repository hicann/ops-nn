# ZZMatMulTOQBMMV3FusionPass

## 融合模式

该融合将MatMul/MatMulV2/MatMulV3、BatchMatMul/BatchMatMulV2/BatchMatMulV3算子转换为QuantBatchMatMulV3算子，并插入scale为1.0的常量作为scale输入，bias输入、转置属性及输出y的数据类型保持不变。

![](../../../docs/zh/figures/ZZMatMulTOQBMMV3FusionPass_1.png)

## 使用约束

- MatMul类型包括BatchMatMul/BatchMatMulV2/MatMul/MatMulV2。
- 输入的数据类型支持HIFLOAT8。
<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持输入为INT8且输出为INT32的组合。
<!-- end id1 -->

## 支持的型号

<!-- npu="950" id2 -->
Ascend 950PR/Ascend 950DT
<!-- end id2 -->
