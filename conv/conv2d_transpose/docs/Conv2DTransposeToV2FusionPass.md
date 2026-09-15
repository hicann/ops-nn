# Conv2DTransposeToV2FusionPass

## 融合模式

该融合将符合图融合pattern的Conv2DTranspose算子改为Conv3DTranspose算子。过程中对输入的filter、x加入Unsqueeze算子，确保Conv3DTranspose的输入维度是5D，对Conv3DTranspose的输出加入Squeeze算子，确保最终输出是4D，符合Conv2DTranspose的输出维度。

融合前：

![](../../../docs/zh/figures/Conv2DTransposeToV2FusionPass_1.png)

融合后：

![](../../../docs/zh/figures/Conv2DTransposeToV2FusionPass_2.png)

## 使用约束

该融合规则不能关闭。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->
