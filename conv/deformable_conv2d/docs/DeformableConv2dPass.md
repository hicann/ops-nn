# DeformableConv2dPass

## 融合模式

该融合规则将DeformableConv2D算子拆分为DeformableOffset算子和Conv2D算子。

![DeformableConv2dPass融合示意图](../../../docs/zh/figures/DeformableConv2dPass_1.png)

## 使用约束

- DeformableConv2D算子的输入个数只能为3（无bias）或4（有bias）。
- filter的格式只支持HWCN或NCHW，且filter维度为4D；filter的H/W（ksize）不能为动态shape。
- fmap和输出的原始shape维度必须为4D。
- strides属性维度必须为4。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->
