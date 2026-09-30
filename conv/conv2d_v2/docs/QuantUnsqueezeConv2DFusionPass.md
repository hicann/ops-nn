# QuantUnsqueezeConv2DFusionPass

## 融合模式

该融合规则将AscendQuant+Unsqueeze+Conv2D+Squeeze+AscendDequant结构调整为Unsqueeze+AscendQuant+Conv2D+AscendDequant+Squeeze。

![QuantUnsqueezeConv2DFusionPass融合示意图](../../../docs/zh/figures/QuantUnsqueezeConv2DFusionPass_1.png)

## 使用约束

- Conv2D的输入节点必须为Unsqueeze，Unsqueeze的输入节点必须为AscendQuant。
- Conv2D的输出节点必须为Squeeze，Squeeze的输出节点必须为AscendDequant。
- AscendQuant、Unsqueeze、Conv2D、Squeeze、AscendDequant节点均不能带控制边。
- Unsqueeze的输入维度必须为3。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->

<!-- npu="A3" id2 -->
Atlas A3系列产品
<!-- end id2 -->
