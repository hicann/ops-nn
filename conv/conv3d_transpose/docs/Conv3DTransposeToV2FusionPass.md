# Conv3DTransposeToV2FusionPass

## 融合模式

该融合将符合图融合pattern的Conv3DTranspose算子改为Conv3DTransposeV2算子。过程中对满足约束条件的filter加入Transpose算子。

融合前：

![](../../../docs/zh/figures/Conv3DTransposeToV2FusionPass_1.png)

融合后：

![](../../../docs/zh/figures/Conv3DTransposeToV2FusionPass_2.png)

## 使用约束

该融合规则在满足以下条件时对format为NCDHW且shape已知的filter新增Transpose算子。

- groups=1。
- filter的数据类型为float32或float16。
- filter的D、H、W维度乘积大于1。
- filter的C维度大于filter的H、W维度乘积，且filter的C维度等于16或不小于32。
- 当filter的C维度大于N维度时，C维度的值小于1.5倍的N维度；当filter的N维度大于C维度时，N维度的值小于1.5倍的C维度。

该融合规则不能关闭。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->
