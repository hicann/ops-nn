# MatMulToMatmulV3FusionPass

## 融合模式

该融合将符合图融合pattern的MatMulV2/MatMul、BatchMatMulV2/BatchMatMul的算子转换为MatMulV3/BatchMatMulV3算子。

![](../../../docs/zh/figures/MatMulToMatmulV3FusionPass_1.png)

## 使用约束

- 输入x1、x2支持的数据类型：FLOAT16、BFLOAT16、FLOAT32、INT8。
- 当x1、x2中一方为INT8，另一方为FLOAT16、BFLOAT16、FLOAT32时，在INT8一侧插入Cast算子，将INT8输入转换为另一侧的数据类型后再接入MatMulV3/BatchMatMulV3，输出y的数据类型与另一侧保持一致。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->
