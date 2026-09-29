# NpuScatterAddBwd（experimental）

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Atlas A3系列产品</term>| √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Ascend 950PR&950DT系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：NpuScatterAddBwd 是 NpuScatterAdd（带缩放因子散射累加，MoE token 聚合）的反向算子，根据索引向量 `indices` 从上游梯度 `y_grad` 中取出对应行，分别计算对源张量 `x` 和缩放因子 `s` 的梯度，常用于 MoE（Mixture of Experts）模型训练中的反向传播。
- 计算公式（对每个 $i = 0, 1, \ldots, N-1$）：

  $$
  x\_grad[i, :] = y\_grad[\text{indices}[i], :] \times s[i]
  $$

  $$
  s\_grad[i] = \sum_{j=0}^{H-1} y\_grad[\text{indices}[i], j] \times x[i, j]
  $$

  其中 `y_grad` 形状为 `(D, H)`，`x` 形状为 `(N, H)`，`s` 形状为 `(N,)`，`indices` 形状为 `(N,)` 且取值范围为 `[0, D)`；输出 `x_grad` 形状为 `(N, H)`，`s_grad` 形状为 `(N,)`。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :---: | :--- | :--- | :---: |
| y_grad | 输入 | 上游梯度张量，形状 (D, H)。 | BFLOAT16、FLOAT16 | ND |
| x | 输入 | 前向中的源张量，形状 (N, H)。 | BFLOAT16、FLOAT16 | ND |
| s | 输入 | 前向中的逐行缩放因子，形状 (N,)。 | 与 x 一致 | ND |
| indices | 输入 | 目标行索引，形状 (N,)，取值范围 [0, D)。 | INT32 | ND |
| x_grad | 输出 | x 的梯度，形状 (N, H)。 | 与 x 一致 | ND |
| s_grad | 输出 | s 的梯度，形状 (N,)。 | 与 s 一致 | ND |

## 约束说明

- `y_grad`、`x`、`s`、`x_grad`、`s_grad` 的 dtype 必须一致（BFLOAT16 或 FLOAT16）。
- `y_grad`、`x`、`x_grad` 的 `dim[1]`（即隐藏维度 H）必须相同。
- `x`、`s`、`indices`、`x_grad`、`s_grad` 的 `dim[0]` 必须相同。
- `y_grad` 和 `x` 必须为 2 维张量，`s` 和 `indices` 必须为 1 维张量。
- 隐藏维度 H 需满足：`align(H) * (element_size * 4 + 4 * 4) + 32 <= 180KB`（受 UB 容量限制，其中 `align(H)` 为 H 按 32 字节对齐后的元素个数，`element_size` 为 2 字节）。
- 当前支持 Atlas A2 和 Atlas A3系列产品。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_npu_scatter_add_bwd.cpp](./examples/test_aclnn_npu_scatter_add_bwd.cpp) | 通过[aclnnNpuScatterAddBwd](./docs/aclnnNpuScatterAddBwd.md)接口调用并校验输出。 |
| PTA调用 | [npu_scatter_add_bwd](./torch_extension/npu_scatter_add_bwd.py) | 通过`torch.ops.cann_ops_nn.npu_scatter_add_bwd(y_grad, x, s, indices)`调用，返回`(x_grad, s_grad)`。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| :--- | :--- | :--- | :--- | :--- |
| Meituan | 美团 | NpuScatterAddBwd | 2026/09/25 | 新增 MoE 场景带缩放因子散射累加算子的反向算子（计算 x 与 s 的梯度）。 |

## 性能数据

以下数据在 Atlas A2（ascend910b，Vector ALU fp32 峰值算力 11 TFLOPS，GM→L1/UB 峰值带宽 1.6 TB/s）上实测，固定 D=16384，H=6144，bf16：

| D | H | N | 耗时(μs) | MFU | MBU |
|---|---|---|----------|-------|-------|
| 16384 | 6144 | 256 | 18 | 2.40% | 33.05% |
| 16384 | 6144 | 512 | 18 | 4.67% | 64.20% |
| 16384 | 6144 | 1024 | 30 | 5.81% | 79.92% |
| 16384 | 6144 | 2048 | 52 | 6.62% | 91.02% |
| 16384 | 6144 | 4096 | 98 | 7.02% | 96.57% |
| 16384 | 6144 | 8192 | 284 | 4.83% | 66.37% |
| 16384 | 6144 | 16384 | 558 | 4.92% | 67.62% |
