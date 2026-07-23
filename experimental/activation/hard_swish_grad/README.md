# HardSwishGrad

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A2 训练系列产品 | √ |
| Atlas A3 系列产品 | √ |

## 功能说明

- 算子功能：计算 HardSwish 激活函数的反向梯度。输入上游梯度 `grad` 和前向输入 `x`，输出 `y`。
- 计算公式：

  ```text
  y = 0                         , x <= -3
  y = grad * (x / 3 + 0.5)      , -3 < x < 3
  y = grad                      , x >= 3
  ```

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :---: | :--- | :--- | :--- |
| grad | 输入 | 上游梯度 Tensor | FLOAT16、FLOAT、BFLOAT16 | ND |
| x | 输入 | HardSwish 前向输入 Tensor，用于确定梯度系数 | 与 `grad` 一致 | 与 `grad` 一致 |
| y | 输出 | HardSwish 反向梯度 Tensor | 与 `grad` 一致 | 与 `grad` 一致 |

## 约束说明

- `grad`、`x` 和 `y` 的数据类型及数据格式必须一致。
- `grad` 和 `x` 的 shape 必须一致，不支持广播。
- `y` 的 shape 与 `grad` 一致。
- 支持动态 shape 和动态 rank。
- 支持空 Tensor。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| aclnn接口 | [test_aclnn_hard_swish_grad](examples/test_aclnn_hard_swish_grad.cpp) | 通过 [aclnnHardswishBackward](docs/aclnnHardswishBackward.md) 接口调用 HardSwishGrad 算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| :--- | :--- | :--- | :--- | :--- |
| [Andy Zhang](https://gitcode.com/hehe7758511) | 西北工业大学智能感知交互实验室 | HardSwishGrad | 2026/7/12 | HardSwishGrad 算子适配开源仓 |
