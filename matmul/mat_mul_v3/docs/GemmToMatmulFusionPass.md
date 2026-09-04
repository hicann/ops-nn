# GemmToMatmulFusionPass

## 融合模式

由于Gemm接口无单算子实现，该融合将符合图融合pattern的Gemm算子拆分成包含matmul、mul、add的几种算子的组合。

![](../../../docs/zh/figures/GemmToMatmulFusionPass_1.png)

## 使用约束

- 输入a和b的dtype支持FLOAT16、FLOAT32、INT32（INT32不支持动态shape）。
- 输出c的dtype支持FLOAT16、FLOAT32、INT32。
- 特殊场景：输入a、b为INT8，输出为FLOAT32。
- 该融合规则不能关闭。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->
