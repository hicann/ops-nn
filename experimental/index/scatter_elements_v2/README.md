# ScatterElementsV2（experimental）

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Atlas A3系列产品</term>| √ |
| <term>Atlas A2系列产品</term> | √ |

## 功能说明

- 算子功能：根据 `indices` 将 `updates` 中的元素写入 `var`，并支持指定轴上的规约计算。
- 本目录同时承载 `aclnnMaxUnpool2d` 的实验性增强实现，使 `self/out` 支持 `BFLOAT16`，`indices` 支持 `INT32` 和 `INT64`。
- 本目录同时承载 `aclnnMaxUnpool3d` 的实验性实现：最大池化的逆过程即"按 `indices` 把输入元素散射到输出"，与 ScatterElementsV2 语义一致，故不新增算子，而是复用本算子实现。
- 针对 `reduction="none"` 且 `var` 末轴长度远大于更新点数的 `BFLOAT16` 稀疏散射场景，新增一条分桶散射分支：先按输出 tile 把更新点归桶，再逐 tile 顺序覆盖，把随机散射写降为顺序突发写，复杂度由 O(numTiles×k) 降为 O(k)。该分支仅在上述条件全部满足时启用，其余场景的选路与既有实现完全一致。
- 当 `reduction="none"` 时，计算关系如下：

  $$
  var[indices[i]] = updates[i]
  $$

  `reduction` 为 `add`、`mul`、`min`、`max` 或 `mean` 时，对相同目标位置执行相应规约。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :---: | :--- | :--- | :---: |
| var | 输入/输出 | 被更新的张量，输出形状与输入一致。 | FLOAT、FLOAT16、DOUBLE、INT32、INT64、UINT8、INT8、INT16、BFLOAT16、BOOL | ND |
| indices | 输入 | 指定 `updates` 中各元素写入 `var` 的目标索引。 | INT32、INT64 | ND |
| updates | 输入 | 用于更新 `var` 的数据。 | 与 `var` 一致 | ND |
| axis | 属性 | 执行索引更新的轴，支持负轴。默认值为 `0`。 | INT64 | - |
| reduction | 属性 | 规约方式，支持 `none`、`add`、`mul`、`min`、`max`、`mean`。默认值为 `none`。 | STRING | - |
| include_self | 属性 | 规约时是否包含 `var` 中的原始值。默认值为 `true`。 | BOOL | - |

## 约束说明

- `var`、`indices` 和 `updates` 的维度数应一致。
- `indices` 与 `updates` 的形状应一致，非 `axis` 轴的大小不得超过 `var` 对应轴的大小。
- `indices` 中的值必须位于 `[-var.shape[axis], var.shape[axis])` 范围内。
- 当前支持 Atlas A2 和 Atlas A3系列产品。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_max_unpool2d.cpp](./examples/test_aclnn_max_unpool2d.cpp) | 使用 BF16 数据和 INT32 索引，通过[aclnnMaxUnpool2d](./docs/aclnnMaxUnpool2d.md)接口调用并校验输出。 |
| aclnn调用 | [test_aclnn_max_unpool3d.cpp](./examples/test_aclnn_max_unpool3d.cpp) | 使用 BF16 数据和 INT32 索引，通过[aclnnMaxUnpool3d](./docs/aclnnMaxUnpool3d.md)接口调用并校验输出。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| :--- | :--- | :--- | :--- | :--- |
| Tream | 个人开发者 | ScatterElementsV2 | 2026/09/20 | 增强 `aclnnMaxUnpool2d`：支持 BFLOAT16 数据及 INT32/INT64 索引。 |
| dududu121 | 个人开发者 | ScatterElementsV2 | 2026/09/23 | 新增 `aclnnMaxUnpool3d`，并为稀疏散射场景增加分桶散射分支。 |
