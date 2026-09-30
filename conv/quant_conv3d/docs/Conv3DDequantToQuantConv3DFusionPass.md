# Conv3DDequantToQuantConv3DFusionPass

## 融合模式

该融合规则将Conv3d+AscendDequant融合为QuantConv3d。

![Conv3D和AscendDequant融合为QuantConv3D](../../../docs/zh/figures/Conv3DDequantToQuantConv3DFusionPass_1.png)

## 使用约束

- 仅支持Conv3d输入输出Dtype为int8输入int32输出。
- 仅支持Conv3d输入输出Format为NCDHW或NDHWC。
- 仅支持Conv3d单输出。
- AscendDequant的relu_flag和sqrt_mode必须为false。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->
