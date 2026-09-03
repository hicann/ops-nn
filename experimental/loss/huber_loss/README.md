# HuberLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品</term>     |     √    |

## 功能说明

HuberLoss 计算 `input` 与 `target` 的 Huber 损失，对齐 PyTorch `aten::huber_loss` 前向语义，支持 `none`/`mean`/`sum` 三种规约模式。

设 $e = input - target$，逐元素损失为：

$$
l =
\begin{cases}
0.5e^2, & |e| \leq delta \\
delta(|e| - 0.5delta), & |e| > delta
\end{cases}
$$

再按 `reduction` 规约：

$$
loss =
\begin{cases}
l, & reduction = 0\ (none) \\
\frac{1}{N}\sum l, & reduction = 1\ (mean) \\
\sum l, & reduction = 2\ (sum)
\end{cases}
$$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | Shape 规格 |
| --- | --- | --- | --- | --- | --- |
| input | 输入 | 预测值张量。 | FLOAT、FLOAT16、BFLOAT16 | ND | 任意维，含 0 维（标量）与空张量。 |
| target | 输入 | 目标值张量。 | 与 `input` 一致 | ND | 必须与 `input` 完全一致。 |
| reduction | 可选属性 | 规约模式，`0`=none、`1`=mean、`2`=sum，默认 `1`。 | INT | - | 取值必须为 `0`/`1`/`2`。 |
| delta | 可选属性 | Huber 分段阈值，默认 `1.0`。 | FLOAT | - | 必须大于 `0`，允许 `+∞`。 |
| output | 输出 | Huber 损失。 | 与 `input` 一致 | ND | `reduction=0` 时与 `input` 一致；否则为 0 维标量。 |

> `reduction` 的取值约定为 **0=none、1=mean、2=sum**，与 PyTorch 一致。同目录的 `smooth_l1_loss_v2` 使用 `1=sum、2=mean` 的相反约定，移植代码时勿直接沿用。

## 约束说明

- `input`、`target` 的 shape 与 dtype 必须完全一致，不支持广播。
- `reduction=0` 时 `output` 的 shape、dtype 与 `input` 一致；`reduction=1/2` 时 `output` 为 0 维标量（兼容 shape 为 `{1}` 的 1 维张量），dtype 与 `input` 一致。
- 参数名为 `input` / `target` / `reduction` / `delta` / `output`。aclnn 两段式接口的输出形参由框架生成，名称恒为 `out`，与此处的 `output` 指同一个张量。
- 支持动态 rank 与动态 shape。
- 支持非连续输入，由框架 `AutoContiguous` 完成连续化。
- FLOAT16、BFLOAT16 在 Kernel 内提升为 FLOAT 计算，末端一次舍入回输出 dtype；`mean`/`sum` 的累加同样在 FLOAT 域完成。
- `reduction=1/2` 使用跨核归约，需要算子自身的 workspace，且启用 `BATCH_MODE` 调度以保证各核共驻。注意第一段接口返回的 `workspaceSize` 还包含框架保留的系统 workspace，三种 `reduction` 下均不为 `0`。
- 空张量：`sum` 返回 `0`，`mean` 返回 `NaN`（`0/0`），与 PyTorch 一致。
- 仅支持表中所列产品。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| ACLNN 调用 | [aclnnHuberLoss 接口文档](./docs/aclnnHuberLoss.md)、[test_aclnn_huber_loss.cpp](./examples/test_aclnn_huber_loss.cpp) | 使用 `aclnnHuberLossGetWorkspaceSize` 和 `aclnnHuberLoss` 两段式接口。 |
| GE | - | 暂不提供。 |
| PyTorch | - | 暂不提供。 |

## 本地编译运行 UT

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
bash build.sh -u --ophost --ops=huber_loss --soc=ascend910b --experimental
bash build.sh -u --opkernel --ops=huber_loss --soc=ascend910b --experimental
```

## 本地自测 UT 覆盖率

- Host InferShape/InferDataType UT：覆盖 `none`/`mean`/`sum` 三种规约的输出 shape（含 0 维标量与空张量）、动态维度，以及 shape/dtype 不一致、`reduction` 越界的失败场景。
- Host Tiling UT：覆盖三种数据类型、属性槽位读取、多核元素均分守恒、tile 随可用 UB 变化、`mean`/`sum` 的 workspace 估算、UB 不足、非法 dtype/shape/`delta`/`reduction` 等失败场景。
- Kernel UT：覆盖 `|e| <= delta` 与 `|e| > delta` 两个分支的手算对拍、非默认 `delta`、不足一个 tile 的尾段。

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| 杨镇泽 | CQUPT | HuberLoss | 2026/08/31 | HuberLoss 算子补齐 reduction（none/mean/sum）支持，适配开源仓 |
