# BatchNormGradV3

本目录包含 BatchNormGradV3 算子的 Ascend C Kernel 实现、Host 侧 tiling 与形状和数据类型推导、ACLNN 接口及图模式注册代码，用于批归一化反向传播。

接口说明见 [aclnnBatchNormBackward](docs/aclnnBatchNormBackward.md) 和 [aclnnFastBatchNormBackward](docs/aclnnFastBatchNormBackward.md)。
