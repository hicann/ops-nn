# Conv2DBackpropFilterToV3FusionPass

## 融合模式

该融合将符合图融合pattern的Conv2DBackpropFilter算子改为Conv3DBackpropFilter算子。过程中对输入的x、out\_backprop加入Unsqueeze算子，确保Conv3DBackpropFilter的输入维度是5D，对Conv3DBackpropFilter的输出加入Squeeze算子，确保最终输出是4D，符合Conv2DBackpropFilter的输出维度。

融合前：

![](../../../docs/zh/figures/Conv2DBackpropFilterToV3FusionPass_1.png)

融合后：

![](../../../docs/zh/figures/Conv2DBackpropFilterToV3FusionPass_2.png)

## 使用约束

该融合规则不能关闭。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->
