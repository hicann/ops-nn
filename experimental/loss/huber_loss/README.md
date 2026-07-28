# HuberLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品</term>     |     √    |

## 功能说明

HuberLoss 计算 `predictions` 与 `targets` 的逐元素 Huber 损失，不执行广播或 reduction。

设 `e = predictions - targets`，则：

$$
loss =
\begin{cases}
0.5e^2, & |e| \leq delta \\
delta(|e| - 0.5delta), & |e| > delta
\end{cases}
$$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | Shape 规格 |
| --- | --- | --- | --- | --- | --- |
| predictions | 输入 | 预测值张量。 | FLOAT、FLOAT16、BFLOAT16 | ND | 必须与 `targets` 完全一致。 |
| targets | 输入 | 目标值张量。 | FLOAT、FLOAT16、BFLOAT16 | ND | 必须与 `predictions` 完全一致。 |
| delta | 可选属性 | Huber 分段阈值，默认 `1.0`。 | FLOAT | - | 必须大于 0。 |
| loss | 输出 | 逐元素 Huber 损失。 | 与 `predictions` 一致 | ND | 与 `predictions` 一致。 |

## 约束说明

- `predictions`、`targets` 和 `loss` 的 shape、dtype 必须一致。
- 支持动态 rank 和动态 shape。
- 不支持广播、reduction 或 workspace。
- FLOAT16、BFLOAT16 在 Kernel 内优先转换为 FLOAT 计算，再转换回输出 dtype。
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

- Host InferShape UT：覆盖普通 shape、标量、空张量、动态 shape，以及输入 shape 不一致的失败场景。
- Host Tiling UT：覆盖 FLOAT、FLOAT16、BFLOAT16、单元素、空张量、非均匀多核切分、多 tile、默认 tiling key、UB 不足、非法 dtype/shape/`delta`。
- Kernel UT：覆盖三种数据类型、`|e| < delta`、`|e| = delta`、`|e| > delta` 两个计算分支及边界、自定义 `delta`、单元素、非均匀多核和多 tile。
- Golden 数据脚本位于 [huber_loss_data](./tests/ut/op_kernel/huber_loss_data)，可生成并比较 FLOAT、FLOAT16、BFLOAT16 数据。

## eager 调用前置条件

先 source 匹配的 CANN 环境，并构建、安装包含该算子的 custom package；运行时使用 `cust`，且 `vendor_name=custom` 必须与安装包匹配。随后执行：

```bash
bash build.sh --run_example huber_loss eager cust --example_name=huber_loss --vendor_name=custom --experimental
```
