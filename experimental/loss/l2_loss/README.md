# L2Loss

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------- |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

- 算子功能：对输入张量 x 的所有元素计算 L2 Loss，即 `output = sum(x * x) / 2`，返回一个标量。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :----- | :-------- | :--- | :------- | :------- |
| x | 输入 | 输入张量 | float32、float16、bfloat16 | ND |
| y | 输出 | 输出标量（shape 为 [1]） | float32、float16、bfloat16 | ND |

## 约束说明

- 本算子验证基于 CANN 9.0.0。
- 输入张量 x 不能为空张量（元素数量必须大于 0）。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :------- | :------- | :--- |
| aclnn 接口 | [test_aclnn_l2_loss](examples/test_aclnn_l2_loss.cpp) | 通过 aclnnL2Loss 接口调用 l2_loss 算子。 |

## 贡献说明

本算子由社区开发者贡献，贡献流程请参考 [CANN 算子贡献指南](https://gitcode.com/cann/ops-math/blob/master/CONTRIBUTING.md)。

欢迎提交 Issue 或 Pull Request 参与共建。如有问题，请在对应仓库的 Issue 区描述复现步骤和环境信息。
