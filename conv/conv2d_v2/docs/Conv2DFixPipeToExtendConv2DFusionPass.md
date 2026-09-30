# Conv2DFixPipeToExtendConv2DFusionPass

## 融合模式

该融合规则将Conv2D+FixPipe融合为ExtendConv2D。

### 单输出融合

![Conv2DFixPipeToExtendConv2DFusionPass融合模式一](../../../docs/zh/figures/Conv2DFixPipeToExtendConv2DFusionPass_1.png)

![Conv2DFixPipeToExtendConv2DFusionPass融合模式二](../../../docs/zh/figures/Conv2DFixPipeToExtendConv2DFusionPass_2.png)

![Conv2DFixPipeToExtendConv2DFusionPass融合模式三](../../../docs/zh/figures/Conv2DFixPipeToExtendConv2DFusionPass_3.png)

### 双输出融合

![Conv2DFixPipeToExtendConv2DFusionPass融合模式四](../../../docs/zh/figures/Conv2DFixPipeToExtendConv2DFusionPass_4.png)

![Conv2DFixPipeToExtendConv2DFusionPass融合模式五](../../../docs/zh/figures/Conv2DFixPipeToExtendConv2DFusionPass_5.png)

![Conv2DFixPipeToExtendConv2DFusionPass融合模式六](../../../docs/zh/figures/Conv2DFixPipeToExtendConv2DFusionPass_6.png)

![Conv2DFixPipeToExtendConv2DFusionPass融合模式七](../../../docs/zh/figures/Conv2DFixPipeToExtendConv2DFusionPass_7.png)

![Conv2DFixPipeToExtendConv2DFusionPass融合模式八](../../../docs/zh/figures/Conv2DFixPipeToExtendConv2DFusionPass_8.png)

![Conv2DFixPipeToExtendConv2DFusionPass融合模式九](../../../docs/zh/figures/Conv2DFixPipeToExtendConv2DFusionPass_9.png)

![Conv2DFixPipeToExtendConv2DFusionPass融合模式十](../../../docs/zh/figures/Conv2DFixPipeToExtendConv2DFusionPass_10.png)

![Conv2DFixPipeToExtendConv2DFusionPass融合模式十一](../../../docs/zh/figures/Conv2DFixPipeToExtendConv2DFusionPass_11.png)

![Conv2DFixPipeToExtendConv2DFusionPass融合模式十二](../../../docs/zh/figures/Conv2DFixPipeToExtendConv2DFusionPass_12.png)

## 使用约束

- 仅支持Conv2D的输入输出Dtypes组合为：int8输入int32输出，或fp16输入fp16输出。
- 仅支持Conv2D的输入输出Formats为NCHW或NHWC。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->
