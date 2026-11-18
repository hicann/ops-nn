# BatchNormGradV3

本目录包含 BatchNormGradV3 算子的 Ascend C Kernel 实现、Host 侧 tiling 与形状和数据类型推导、ACLNN 接口及图模式注册代码，用于批归一化反向传播。

接口说明见 [aclnnBatchNormBackward](docs/aclnnBatchNormBackward.md) 和 [aclnnFastBatchNormBackward](docs/aclnnFastBatchNormBackward.md)。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn调用 | [test_aclnn_batch_norm_backward.cpp](./examples/test_aclnn_batch_norm_backward.cpp) | 通过[aclnnBatchNormBackward](./docs/aclnnBatchNormBackward.md)接口方式调用batch_norm_grad_v3算子。 |
| aclnn调用 | [test_aclnn_fast_batch_norm_backward.cpp](./examples/arch22/test_aclnn_fast_batch_norm_backward.cpp) | 通过[aclnnFastBatchNormBackward](./docs/aclnnFastBatchNormBackward.md)接口方式调用batch_norm_grad_v3算子。 |
