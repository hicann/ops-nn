# NpuScatterAdd（experimental）

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：基于预排序索引的带缩放因子散射累加（MoE token 聚合）。根据索引向量 `indices` 将源张量 `x` 的行按缩放因子 `s` 加权后累加到目标张量 `y` 的对应行上（in-place），常用于 Mixture of Experts 模型中将专家输出按路由索引聚合回原始 token 位置。
- 计算公式（提供缩放因子 `s` 时）：

  $$
  y[\text{indices}[i], :] \mathrel{+}= x[i, :] \times s[i], \quad i = 0, 1, \ldots, S-1
  $$

  不提供缩放因子时：

  $$
  y[\text{indices}[i], :] \mathrel{+}= x[i, :], \quad i = 0, 1, \ldots, S-1
  $$

  其中 `x` 形状为 `(S, H)`，`y` 形状为 `(D, H)` 且同时为输出，`s` 形状为 `(S,)`，`indices` 形状为 `(S,)` 且取值范围为 `[0, D)`。
- kernel 按 `sort_idx`（`argsort(indices)`）顺序遍历源行，将同一目标行的累加在 workspace 中连续完成，减少对 `y` 的重复 HBM 读写，在冲突较少时可显著降低访存量。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :---: | :--- | :--- | :---: |
| x | 输入 | 源张量，形状 (S, H)。 | BFLOAT16、FLOAT16 | ND |
| y | 输入/输出 | 目标张量，形状 (D, H)，计算结果 in-place 累加写入。 | 与 x 一致 | ND |
| s | 输入（可选） | 逐行缩放因子，形状 (S,)。 | 与 x 一致 | ND |
| indices | 输入 | 目标行索引，形状 (S,)，取值范围 [0, D)。 | INT32 | ND |
| sort_idx | 输入 | `argsort(indices)` 的结果，形状 (S,)。 | INT32 | ND |
| valid_token_num | 输入（可选） | 有效 token 数量，形状 (1,)。提供时仅处理前 `valid_token_num` 个行。 | INT32 | ND |
| use_high_precision | 属性 | 是否使用高精度模式（FP32 下完成缩放和累加后再转回原精度）。默认 `false`。 | BOOL | - |

## 约束说明

- `x`、`y` 的 dtype 必须一致（BFLOAT16 或 FLOAT16），输出与 `y` 一致。
- `x`、`y` 的 `dim[1]`（即隐藏维度 H）必须相同。
- `x`、`indices`、`sort_idx` 的 `dim[0]` 必须相同。
- 若提供 `s`，其 dtype 必须与 `x` 一致，且 `dim[0]` 等于 `x` 的 `dim[0]`。
- `x` 和 `y` 必须为 2 维张量，`indices`、`sort_idx` 必须为 1 维张量。
- 隐藏维度 H 需满足：`align(H) * (element_size * 3 + 4 * 3) + 32 <= 180KB`（受 UB 容量限制，其中 `align(H)` 为 H 按 32 字节对齐后的元素个数，`element_size` 为 2 字节）。
- `sort_idx` 必须是 `indices` 的 argsort 结果，即 `sort_idx = argsort(indices)`。
- 当前支持 Atlas A2 和 Atlas A3 系列产品。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_npu_scatter_add.cpp](./examples/test_aclnn_npu_scatter_add.cpp) | 通过[aclnnNpuScatterAdd](./docs/aclnnNpuScatterAdd.md)接口调用并校验输出。 |
| PTA调用 | [npu_scatter_add](./torch_extension/npu_scatter_add.py) | 通过`torch.ops.cann_ops_nn.npu_scatter_add(x, y, indices, sort_idx)`调用，结果 in-place 累加写入`y`并返回`y`。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| :--- | :--- | :--- | :--- | :--- |
| Meituan | 美团 | NpuScatterAdd | 2026/09/25 | 新增 MoE 场景带缩放因子的散射累加算子（含预排序索引加速与高精度模式）。 |
