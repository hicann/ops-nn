# DepthwiseDfFusionPass

## 融合模式

该融合规则为DepthwiseConv2DBackpropInput卷积算子添加Transpose、Reshape算子，将DepthwiseConv2DBackpropInput的filter输入转换成符合Conv2DBackpropInput的格式，调用Conv2DBackpropInput算子。

融合前：

![](../../../docs/zh/figures/DepthwiseDfFusionPass_1.png)

融合后：

![](../../../docs/zh/figures/DepthwiseDfFusionPass_2.png)

## 使用约束

该融合规则会在filter输入格式为NCHW时，依次添加Transpose、Reshape算子，filter输入格式为HWCN时，添加Reshape算子。

该融合规则不能关闭。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->
