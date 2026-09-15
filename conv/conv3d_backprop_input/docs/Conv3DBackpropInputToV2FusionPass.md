# Conv3DBackpropInputToV2FusionPass

## 融合模式

该融合将符合图融合pattern的Conv3DBackpropInput算子改为Conv3DBackpropInputV2算子。过程中对满足约束条件的filter（DHWCN转NCDHW）、out\_backprop（NDHWC转NCDHW）加入Transpose算子，使Conv3DBackpropInputV2以NCDHW格式计算；算子输出侧加入Transpose（NCDHW转回原format），保持图语义不变。转为NCDHW格式是为了命中算子针对NCDHW输入的优化路径，提升该场景性能。

融合前：

![](../../../docs/zh/figures/Conv3DBackpropInputToV2FusionPass_1.png)

融合后：

![](../../../docs/zh/figures/Conv3DBackpropInputToV2FusionPass_2.png)

## 使用约束

该融合规则在满足以下条件时新增Transpose算子：

- filter的format为DHWCN，out\_backprop的format为NDHWC
- strideD、strideH、strideW分别为1、2、2。
- dilationD、dilationH、dilationW分别为1、1、1。
- groups为1。
- 输出、filter、out\_backprop的shape均在固定白名单内。

该融合规则不能关闭。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->
