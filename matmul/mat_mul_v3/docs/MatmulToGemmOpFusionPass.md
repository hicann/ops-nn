# MatmulToGemmOpFusionPass

## 融合模式

<!-- npu="910b" id1 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：该融合将符合图融合pattern的MatMulV3/MatMulV2/MatMul的算子转换为GemmV2算子。
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该融合将符合图融合pattern的MatMulV3/MatMulV2/MatMul的算子转换为GemmV2算子。
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950PR/Ascend 950DT</term>：该融合将符合图融合pattern的MatMulV3/MatMulV2/MatMul的算子转换为GemmV3算子。
<!-- end id3 -->

![](../../../docs/zh/figures/MatmulToGemmOpFusionPass_1.png)

## 使用约束

- MatMul和AssignAdd节点之间的Cast节点可以不存在，支持不带Cast节点的匹配。
- MatMul类型包括MatMul/MatMulV2/MatMulV3。
- MatMul节点的输入dtype：
  <!-- npu="910b" id4 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：仅支持float16和bfloat16。
  <!-- end id4 -->
  <!-- npu="A3" id5 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持float16和bfloat16。
  <!-- end id5 -->
  <!-- npu="950" id6 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：支持float16、float32和bfloat16。
  <!-- end id6 -->
- MatMul节点的输入shape：
  <!-- npu="910b" id7 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：仅支持白名单中的shape进行融合。
  <!-- end id7 -->
  <!-- npu="A3" id8 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持白名单中的shape进行融合。
  <!-- end id8 -->
- AssignAdd节点输入dtype仅支持float32。
- 不建议关闭，关闭后可能会影响网络精度。

## 支持的型号

<!-- npu="910b" id9 -->
Atlas A2 训练系列产品/Atlas A2 推理系列产品
<!-- end id9 -->

<!-- npu="A3" id10 -->
Atlas A3 训练系列产品/Atlas A3 推理系列产品
<!-- end id10 -->

<!-- npu="950" id11 -->
Ascend 950PR/Ascend 950DT
<!-- end id11 -->
