# MatMulBiasAddOpenFusionPass

## 融合模式

将MatMul/MatMulV2/BatchMatMul/BatchMatMulV2算子和BiasAdd/Add算子融合为MatMul/MatMulV2/BatchMatMulV2算子。

![](../../../docs/zh/figures/MatMulBiasAddOpenFusionPass_1.png)

## 使用约束

- MatMul节点的输出只能连接一个下游算子。
- bias只支持一维：Add算子的两个输入中，MatMul输出之外的那个输入（即bias）维度必须为1，不支持多维bias。
- bias的数值大小与MatMul或MatMulV2/BatchMatMul/BatchMatMulV2输出的倒数第一维的数值大小保持一致，不支持broadcast。
- MatMul/MatMulV2的输出维度为2。
- 如果输入为BatchMatMul，融合后转为BatchMatMulV2。
- 如果融合前的BiasAdd/Add是FLOAT16计算，那么融合之后bias会在MatMul内部使用FLOAT32相加，导致精度提升。

## 支持的型号

<!-- npu="950" id1 -->

Ascend 950PR/Ascend 950DT

<!-- end id1 -->
