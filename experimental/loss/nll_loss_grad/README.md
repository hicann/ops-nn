# NllLossGrad

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | ✓ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | ✓ |

## 功能说明

- **算子功能:** 计算负对数似然损失(NLL Loss)关于输入的梯度。对于输入张量x(N,C)、目标target、权重weight和上游梯度y_grad，计算输出梯度x_grad。
- 计算公式：

$$
x\_grad[n][i] =
\begin{cases}
    \displaystyle -y\_grad[n] \cdot \text{weight}, & \text{if } i = \text{target} \text{ 且 } \text{reduction} = \text{none} \\
    \displaystyle -\frac{y\_grad}{\text{total\_weight}} \cdot \text{weight}, & \text{if } i = \text{target} \text{ 且 } \text{reduction} = \text{mean} \\
    \displaystyle -y\_grad \cdot \text{weight}, & \text{if } i = \text{target} \text{ 且 } \text{reduction} = \text{sum} \\
    0, & \text{if } i \neq \text{target}
\end{cases}
$$
## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|---|---|---|---|---|
| x | 输入 | 前向输入张量，shape为(N,C)或(C,) | FLOAT, FLOAT16, BFLOAT16 | ND |
| y_grad | 输入 | 上游梯度 | 与x相同 | ND |
| target | 输入 | 目标类别索引 | INT32, INT64, UINT8 | ND |
| weight | 输入 | 各类别权重，shape为(C,) | 与x相同 | ND |
| total_weight | 输入 | 权重总和标量 | 与x相同 | ND |
| reduction | 属性 | 规约方式："none"/"sum"/"mean" | STRING | - |
| ignore_index | 属性 | 忽略的目标索引值 | INT64 | - |
| x_grad | 输出 | 输入梯度，shape与x相同 | 与x相同 | ND |

## 约束说明

- target中的值必须在[0, C)范围内或等于ignore_index。
- weight的长度必须等于C。
- 当reduction为"mean"时，total_weight不应为0（为0时梯度输出0）。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|---|---|---|
| aclnn接口 | [test_aclnn_nll_loss_grad](examples/test_aclnn_nll_loss_grad.cpp) | 通过 [aclnnNLLLossBackward](docs/aclnnNLLLossBackward.md) 接口调用 |
| aclnn接口 | [test_aclnn_nll_loss_grad_2d](examples/test_aclnn_nll_loss_grad_2d.cpp) | 通过 [aclnnNLLLoss2dBackward](docs/aclnnNLLLoss2dBackward.md) 接口调用 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|---|---|---|---|---|
| [GMOW](https://gitcode.com/gcw_8p1hhlB0) | 西北工业大学智能感知交互实验室  | NllLossGrad | 2026/7 | NllLossGrad算子适配开源仓 |
