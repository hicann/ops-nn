# Conv3DBackpropFilterToV2FusionPass

## 融合模式

该融合将符合图融合pattern的Conv3DBackpropFilter算子改为Conv3DBackpropFilterV2算子。过程中对满足约束条件的输出加入Transpose算子。

融合前：

融合前：

![](../../../docs/zh/figures/Conv3DBackpropFilterToV2FusionPass_1.png)

融合后：

![](../../../docs/zh/figures/Conv3DBackpropFilterToV2FusionPass_2.png)

## 使用约束

- 该融合规则在满足以下条件时对format为NDHWC/DHWCN且shape已知的输出新增Transpose算子：
  - Din等于1时，Din\*Cout\*Cin不大于核数/2\*32\*32。
  - Din大于1时，Din\*Cout\*Cin不大于核数\*32\*32\*4。
- 该融合规则不能关闭。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->
