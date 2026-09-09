# MatMulUnsqueezeSqueezeFusionPass

## 融合模式

**融合模式一**

MatMul/MatMulV2/BatchMatMul/BatchMatMulV2支持1 dim x N dim和N dim x 1 dim的输入场景下，需要将1 dim的输入插入Unsqueeze算子扩成二维，输出插入Squeeze算子去掉对应的扩维轴。

![](../../../docs/zh/figures/MatMulUnsqueezeSqueezeFusionPass_1.png)

**融合模式二**

MatMul/MatMulV2/BatchMatMul/BatchMatMulV2支持1 dim x 1 dim的输入场景下，需要将1 dim的输入插入Unsqueeze算子扩成二维。

![](../../../docs/zh/figures/MatMulUnsqueezeSqueezeFusionPass_2.png)

**融合模式三**

MatMul/MatMulV2/BatchMatMul/BatchMatMulV2支持AscendDequant场景，1维输入的扩维处理与模式一/模式二一致，Squeeze算子插入在AscendDequant算子之后。

![](../../../docs/zh/figures/MatMulUnsqueezeSqueezeFusionPass_3.png)

## 使用约束

- x1、x2中至少一个的维度为1。
- AscendDequant场景下，MatMul的输出只能连接一个下游算子，且该算子须为AscendDequant。

## 支持的型号

<!-- npu="910b" id1 -->

Atlas A2 训练系列产品/Atlas A2 推理系列产品

<!-- end id1 -->

<!-- npu="A3" id2 -->

Atlas A3 训练系列产品/Atlas A3 推理系列产品

<!-- end id2 -->

<!-- npu="950" id3 -->

Ascend 950PR/Ascend 950DT

<!-- end id3 -->
