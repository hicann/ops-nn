# BatchMatMul2MulFusionPass

## 融合模式

网络中存在部分k=1的MatMul/MatMulv2/MatMulV3/BatchMatmul/BatchMatMulV2/BatchMatMulV3算子，性能表现较差。该图融合就是将MatMul/MatMulv2/MatMulV3/BatchMatmul/BatchMatMulV2/BatchMatMulV3转为mul，解决性能问题。

>[!NOTE]说明
>Ascend 950PR/Ascend 950DT仅融合MatMul/MatMulv2/MatMulV3节点，不融合BatchMatmul/BatchMatMulV2/BatchMatMulV3节点。

![](../../../docs/zh/figures/BatchMatMul2MulFusionPass_1.png)

若输入的对应的adj为true则需要在对应输入前插入reshape算子。

![](../../../docs/zh/figures/BatchMatMul2MulFusionPass_2.png)

## 使用约束

- 仅适用于静态场景，输入不带bias。
- DType符合如下条件：
  - 输入输出均为Float32。
  - 输入输出均为Float16，BFloat16。

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
