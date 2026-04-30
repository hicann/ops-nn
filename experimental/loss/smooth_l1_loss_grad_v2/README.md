# SmoothL1LossGradV2

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- **算子功能**：平滑 L1 损失反向传播计算，根据预测值 `predict`、标签值 `label` 以及上游梯度 `dout` 计算对预测值的梯度。支持 `none`、`mean`、`sum` 三种 reduction 模式（归约语义由上层封装完成）。

- **计算公式**：

  设输入为 `predict`、`label` 与上游梯度 `dout`，定义：

  $$
  diff = predict - label
  $$

  梯度因子 `grad_factor` 按分段函数计算：

  $$
  grad\_factor =
  \begin{cases}
  -1.0, & diff \le -\sigma \\
   1.0, & diff \ge  \sigma \\
   \frac{diff}{\sigma}, & |diff| < \sigma
  \end{cases}
  $$

  最终输出梯度为：

  $$
  gradient = dout \times grad\_factor
  $$

  若 `reduction = mean`，则额外乘以缩放因子 $1/N$，其中 $N$ 为 `predict` 的总元素个数。

- **reduction 说明**：
  - `none`：返回逐元素梯度，不进行额外缩放。
  - `mean`：返回缩放后的梯度（在 kernel 内部完成 $1/N$ 乘法）。
  - `sum`：kernel 内部按 `none` 处理，归约操作由上层 aclnn 封装通过 `ReduceSum` 算子完成。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| predict | 输入 | 预测值张量，公式中的 predict。 | BFLOAT16、FLOAT16、FLOAT | ND |
| label | 输入 | 标签张量，公式中的 label。 | BFLOAT16、FLOAT16、FLOAT | ND |
| dout | 输入 | 上游梯度张量，可为标量（shape 为空或仅含一个元素）或与 predict 同 shape 的张量。 | BFLOAT16、FLOAT16、FLOAT | ND |
| sigma | 可选属性 | 平滑阈值，控制二次段与一次段分界，默认值为 1.0。aclnn 接口参数名为 beta，与算子属性 sigma 语义一致。 | FLOAT | — |
| reduction | 可选属性 | 支持 none、mean、sum。none 返回逐元素梯度；mean 返回缩放后梯度；sum 由上层封装完成归约。 | STRING / INT64（接口映射） | — |
| gradient | 输出 | 公式中的输出梯度，shape 与 predict 一致。 | BFLOAT16、FLOAT16、FLOAT | ND |

## 约束说明

- 输入维度范围 1-8 维。
- `predict` 与 `label` 的 dtype 需一致，且 shape 必须完全相同（当前实现不支持广播）。
- `dout` 可为标量或与 `predict` 同 shape 的张量；若为标量，则广播至 `predict` 的 shape。
- `sigma` 必须为正数，若传入非正值，算子内部会使用极小值 `1e-6` 保护除零操作。
- 输出 `gradient` 的 dtype 与 `predict` 保持一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn 调用 | [test_aclnn_smooth_l1_loss_grad_v2.cpp](./examples/test_aclnn_smooth_l1_loss_grad_v2.cpp) | 通过 aclnnSmoothL1LossGradV2 文档说明的两段式接口调用该算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| :--- | :--- | :--- | :--- | :--- |
| yanoyano | 浙江工业大学-智能计算研究所 | SmoothL1LossGradV2 | 2026/04/21 | SmoothL1LossGradV2 算子适配开源仓 |