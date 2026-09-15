# DepthwiseDwMulFusionPass

## 融合模式

该融合规则将DepthwiseConv2DBackpropFilter卷积算子转换为对Conv2DBackpropFilter算子的调用，并对输出依次添加Reshape、Transpose算子，将Conv2DBackpropFilter的输出转换成DepthwiseConv2DBackpropFilter对应的输出格式。

融合前：

![](../../../docs/zh/figures/DepthwiseDwMulFusionPass_1.png)

融合后：

![](../../../docs/zh/figures/DepthwiseDwMulFusionPass_2.png)

## 使用约束

该融合在filter format为NCHW时依次添加Reshape、Transpose算子，filter format为HWCN时添加Reshape算子。

该融合规则不能关闭。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->
