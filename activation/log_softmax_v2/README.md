# LogsoftmaxV2

本目录仅包含LogsoftmaxV2算子对应的aclnn接口；如您想要贡献该算子的AscendC实现，请参考[贡献流程](../../CONTRIBUTING.md)。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn调用 | [test_aclnn_log_softmax_v2.cpp](./examples/test_aclnn_log_softmax_v2.cpp) | 通过[aclnnLogSoftmax](./docs/aclnnLogSoftmax.md)接口方式调用log_softmax_v2算子。 |
| 图模式调用 | [test_geir_log_softmax_v2.cpp](./examples/test_geir_log_softmax_v2.cpp) | 通过[算子IR](./op_graph/log_softmax_v2_proto.h)构图方式调用log_softmax_v2算子。 |
