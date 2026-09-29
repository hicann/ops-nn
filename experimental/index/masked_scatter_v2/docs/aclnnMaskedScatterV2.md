# aclnnMaskedScatterV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | ------- |
| Ascend 950PR（本 v2 新增） | √ |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | ×（请使用 experimental/index/masked_scatter） |

## 功能说明

- **算子功能**：根据布尔掩码 mask，将输入 x 中 mask 为 true 的位置，按扁平顺序替换为 updates 中的值。
- **计算公式**：记 cnt(i) = sum_{k<i} [mask_k = true]，

  y_i = updates_{cnt(i)} 若 mask_i = true；否则 y_i = x_i。

- **与 A2 版（masked_scatter）的差异**：
  1. 面向 Ascend 950PR 的两阶段全向量化实现（PhaseA 计数 → 跨核计数广播 → 向量前缀 + Gather + Select），
     mask 仅读取 2 遍（A2 版为跨核重扫 + 标量 SetValue 路径）；
  2. 支持 mask 相对 x 的**右对齐广播**（限制见约束说明第 4 条）；
  3. 首批支持 dtype：FLOAT、FLOAT16、BFLOAT16、INT32（其余 dtype 为后续扩展项）。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| ------ | -------- | ---- | ------- | ------- |
| x | 输入 | 待更新张量 | FLOAT、FLOAT16、BFLOAT16、INT32 | ND |
| mask | 输入 | 布尔掩码 | BOOL | ND |
| updates | 输入 | 按 mask=true 位置顺序消耗的数据 | 与 x 相同 | ND |
| out | 输出 | 结果张量（shape = x 的 shape） | 与 x 相同 | ND |

## 约束说明

1. mask 广播：mask 的 rank ≤ x 的 rank 且各维可广播（右对齐）；x 元素数 ≥ 2^31 时不支持（int32 偏移域）；
2. 非广播 / 满足"rank ≤ 8 且 mask 最内扩展维 stride = 1 且 x 元素数 ≥ 8192 且 mask 物理字节数 ≥ 32"的广播由 kernel 原生处理；
   其余广播 shape 需调用方先展开为与 x 同 shape（native 准入不满足时 tiling 返回失败并提示）；
3. x、updates、out 数据类型必须相同，且均需 contiguous；
4. updates 按 mask = true 的扁平顺序依次消耗，长度必须 ≥ 广播展开后 mask 的 true 元素数。

## 调用说明

两段式接口：

- `aclnnMaskedScatterV2GetWorkspaceSize(self, mask, updates, out, &workspaceSize, &executor)`
- `aclnnMaskedScatterV2(workspace, workspaceSize, executor, stream)`

参考 `examples/test_aclnn_masked_scatter_v2.cpp`。
