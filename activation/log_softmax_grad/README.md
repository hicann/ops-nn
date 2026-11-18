# LogsoftmaxGrad

本目录仅包含LogsoftmaxGrad算子对应的aclnn接口；如您想要贡献该算子的AscendC实现，请参考[贡献流程](../../CONTRIBUTING.md)。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn调用 | [test_aclnn_log_softmax_grad.cpp](./examples/test_aclnn_log_softmax_grad.cpp) | 通过[aclnnLogSoftmaxBackward](./docs/aclnnLogSoftmaxBackward.md)接口方式调用log_softmax_grad算子。 |
