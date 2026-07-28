# GaussianNllLossGrad

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | √ |

## 功能说明

计算 GaussianNLLLoss 对 `input` 和 `var` 的梯度，不计算 `target` 的梯度。设
`d = input - target`、`v = max(var, eps)`，则每个逻辑元素的梯度为：

```text
gradInput = gradOutput * d / v
gradVar = gradOutput * 0.5 * (1 / v - d² / v²)
```

`reduction="mean"` 时两项额外乘以 `1/N`，其中 `N` 为 `input` 的逻辑元素数。
`reduction="sum"` 和 `"mean"` 接收标量 `gradOutput`，`"none"` 接收与 `input`
同 shape 的 `gradOutput`。`var` 广播时，`gradVar` 归约回原始 `var` shape。
`full` 不影响梯度，仅用于保持前后向接口一致。FLOAT16 和 BFLOAT16 输入转为
FLOAT 计算，再转换回输入 dtype。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | Shape 规格 |
| --- | --- | --- | --- | --- | --- |
| gradOutput | 输入 | 上游梯度。 | FLOAT、FLOAT16、BFLOAT16 | ND | `none` 时与 `input` 相同；`sum`/`mean` 时为单元素标量。 |
| input | 输入 | Gaussian 分布均值预测。 | 与 `gradOutput` 相同 | ND | 任意维静态 shape。 |
| target | 输入 | 目标值。 | 与 `gradOutput` 相同 | ND | 与 `input` 相同，或同 rank 且恰有一个广播维为 1。 |
| var | 输入 | 方差。 | 与 `gradOutput` 相同 | ND | 与 `input` 相同、最后一维为 1、缺少最后一维，或单元素标量。 |
| full | 属性 | 是否包含完整高斯常数项；不影响梯度。默认 `false`。 | BOOL | - | - |
| eps | 属性 | 方差下限。默认 `1e-6`。 | FLOAT | - | 必须大于 0。 |
| reduction | 属性 | 规约方式，默认 `"mean"`。 | STRING | - | `"none"`、`"mean"` 或 `"sum"`。 |
| gradInput | 输出 | 对 `input` 的梯度。 | 与 `gradOutput` 相同 | ND | 与 `input` 相同。 |
| gradVar | 输出 | 对 `var` 的梯度；广播贡献已归约。 | 与 `gradOutput` 相同 | ND | 与 `var` 相同。 |

## 约束说明

- 仅支持 Atlas A2、ND，以及 FLOAT、FLOAT16、BFLOAT16。
- 所有输入与输出 dtype 必须一致，不创建 dtype-only tiling key。
- `target` 仅允许同 shape，或在一个维度上从 1 广播到 `input`。
- `var` 仅允许同 shape、最后一维为 1、缺少最后一维或单元素标量。
- `eps` 必须大于 0；`var` 的值约束为非负。
- clamp 对梯度透明：即使 `var < eps`，仍按 `v=max(var, eps)` 的公式计算 `gradVar`。
- `full` 不改变 `gradInput` 或 `gradVar`。
- 空 `input` 不产生 `gradInput` 元素；存在单元素 `var` 时 `gradVar` 为 0。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| ACLNN 调用 | [接口文档](./docs/aclnnGaussianNllLossGrad.md)、[样例](./examples/test_aclnn_gaussian_nll_loss_grad.cpp) | 两段式接口。 |
| GE | - | 暂不提供。 |
| PyTorch | - | 暂不提供。 |

## 本地编译运行 UT

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
bash build.sh -u --ophost --ops=gaussian_nll_loss_grad --soc=ascend910b --experimental
bash build.sh -u --opkernel --ops=gaussian_nll_loss_grad --soc=ascend910b --experimental
```

## 本地自测 UT 覆盖率

普通 UT 通过且已安装 `lcov` 后执行：

```bash
bash build.sh -u --ophost --ops=gaussian_nll_loss_grad --soc=ascend910b --experimental --cov
bash build.sh -u --opkernel --ops=gaussian_nll_loss_grad --soc=ascend910b --experimental --cov
```

## eager 调用前置条件

先 source 匹配的 CANN 环境，使用 `--experimental` 构建并安装最新 custom package，
再以 `cust` 和匹配的 `vendor_name=custom` 运行样例。

## 参考资源

- [PyTorch GaussianNLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html)
