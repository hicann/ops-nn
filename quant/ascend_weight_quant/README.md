# AscendWeightQuant

[English](./README_en.md) | 简体中文

如您想要贡献该算子的AscendC实现，请参考[贡献流程](../../CONTRIBUTING.md)。

## 功能说明

量化链路中的权重量化算子（opFile=Null，无 kernel 实现，仅由融合 pass 在图编译期处理）。

- 输入：`x`（int8，HWCN，必选）、`offset`（int8，HWCN，必选）
- 输出：`y`（int8，HWCN，必选）
- 属性：`dst_type`（可选 int，支持 `DT_INT8`/`DT_INT4`，默认 `DT_INT8`）
- 支持芯片：ascend950 / ascend960dt / ascend960pr
- aclnn 侧：aclnn_exclude（不暴露 aclnn 接口）
