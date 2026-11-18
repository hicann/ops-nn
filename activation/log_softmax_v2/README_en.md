# LogsoftmaxV2

English | [简体中文](./README.md)

This directory only contains the aclnn interface corresponding to the LogsoftmaxV2 operator. If you want to contribute the AscendC implementation of this operator, refer to [contribution process](../../CONTRIBUTING_en.md).

## Invocation Description

| Invocation Method | Sample Code | Description |
| --- | --- | --- |
| aclnn invocation | [test\_aclnn\_log\_softmax\_v2](./examples/test_aclnn_log_softmax_v2.cpp) | Invokes the log_softmax_v2 operator through the [aclnnLogSoftmax](./docs/aclnnLogSoftmax_en.md) APIs |
| Graph-mode invocation | [test\_geir\_log\_softmax\_v2](./examples/test_geir_log_softmax_v2.cpp) | Invokes the log_softmax_v2 operator by building a graph through the [operator IR](./op_graph/log_softmax_v2_proto.h) |
