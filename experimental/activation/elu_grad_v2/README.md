# EluGradV2

## 产品支持情况

| 产品 | 是否支持 |
|:--|:--:|
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 支持 |

> 当前算子代码已注册 `ascend910b` AICore 配置。

## 功能说明

- 算子功能：完成 ELU 激活函数的反向梯度计算。
- 输入含义：
  - `dy` 表示上游梯度；
  - `x` 在 `is_result=false` 时表示 ELU 前向输入，在 `is_result=true` 时表示 ELU 前向输出。

- 计算公式：

当 `x > 0` 时：

$$
y = dy \times scale
$$

当 `x ≤ 0` 且 `is_result = false` 时：

$$
y = dy \times \alpha \times scale \times input\_scale \times \exp(x \times input\_scale)
$$

当 `x ≤ 0` 且 `is_result = true` 时：

$$
y = dy \times input\_scale \times (x + \alpha \times scale)
$$

- 实现说明：
  - Host 侧完成算子定义、shape/dtype 推导和 tiling 计算；
  - Kernel 侧按 `schMode` 分发到 generic 路径、single-tile 路径和对齐 fast path；
  - 对 `float16` / `bfloat16` 通用路径，内部通过提升到 `float32` 计算来兼顾精度。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:--|:--:|:--|:--|:--|
| `dy` | 输入 | ELU 反向计算的上游梯度 | `FLOAT`、`FLOAT16`、`BFLOAT16` | `ND` |
| `x` | 输入 | 当 `is_result=false` 时表示 ELU 前向输入；当 `is_result=true` 时表示 ELU 前向输出 | `FLOAT`、`FLOAT16`、`BFLOAT16`，且需与 `dy` dtype 一致 | `ND` |
| `y` | 输出 | ELU 反向计算结果，即前向输入对应的梯度 | 通常与 `dy` 同 dtype；底层实现支持部分 mixed-output 场景 | `ND` |
| `alpha` | 属性 | ELU 激活系数 | `float` | - |
| `scale` | 属性 | 输出缩放系数 | `float` | - |
| `input_scale` | 属性 | 输入缩放系数 | `float` | - |
| `is_result` | 属性 | `false` 表示 `x` 为前向输入，`true` 表示 `x` 为前向输出 | `bool` | - |

## 约束说明

- `dy` 与 `x` 的 dtype 必须一致，当前支持 `float16`、`float32`、`bfloat16`。
- 输出 `y` 的 shape 必须与输入一致。
- 推荐输入输出使用 `ND` 格式。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:--|:--|:--|
| `aclnn` 直调 | [test_aclnn_elu_grad_v2.cpp](./examples/test_aclnn_elu_grad_v2.cpp) | 通过[aclnnEluV2](./docs/aclnnEluBackward.md)接口方式调用EluV2算子 |


## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| 何洋洋 | 北京交通大学-赵宏智老师团队 | EluGradV2 | 2026/06/05 | EluGradV2算子适配开源仓 |
