# BatchNormGradV3

English | [简体中文](./README.md)

This directory contains the Ascend C Kernel implementation of the BatchNormGradV3 operator, the Host-side tiling and shape/data type inference, the ACLNN interface, and the graph-mode registration code, used for batch normalization backpropagation.

For the interface descriptions, see [aclnnBatchNormBackward](docs/aclnnBatchNormBackward_en.md) and [aclnnFastBatchNormBackward](docs/aclnnFastBatchNormBackward_en.md).

## Invocation Description

| Invocation Method | Sample Code | Description |
| --- | --- | --- |
| aclnn invocation | [test\_aclnn\_batch\_norm\_backward](./examples/test_aclnn_batch_norm_backward.cpp) | Invokes the batch_norm_grad_v3 operator through the [aclnnBatchNormBackward](./docs/aclnnBatchNormBackward_en.md) APIs |
| aclnn invocation | [test\_aclnn\_fast\_batch\_norm\_backward](./examples/arch22/test_aclnn_fast_batch_norm_backward.cpp) | Invokes the batch_norm_grad_v3 operator through the [aclnnFastBatchNormBackward](./docs/aclnnFastBatchNormBackward_en.md) APIs |
