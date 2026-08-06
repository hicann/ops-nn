# aclnnEmbeddingDenseGrad


## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas 310P 推理系列产品 | √ |
## 功能说明

- 接口功能：计算 `Embedding` 算子的反向梯度。给定前向输出对 `grad` 的梯度以及索引 `indices`，按 `scatter-add` 语义将 `grad` 的每一行累加到输出 `y` 中 `indices[i]` 指定的行上。
- 当前算子提供的 ACLNN 接口为 `aclnnEmbeddingDenseGradGetWorkspaceSize` 和 `aclnnEmbeddingDenseGrad` 两段式接口。

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnEmbeddingDenseGradGetWorkspaceSize` 获取计算所需 workspace 大小以及执行器，再调用 `aclnnEmbeddingDenseGrad` 执行计算。

```Cpp
aclnnStatus aclnnEmbeddingDenseGradGetWorkspaceSize(
    const aclTensor* grad,
    const aclTensor* indices,
    int64_t numWeights,
    int64_t paddingIdx,
    bool scaleGradByFreq,
    const aclTensor* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

```Cpp
aclnnStatus aclnnEmbeddingDenseGrad(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## aclnnEmbeddingDenseGradGetWorkspaceSize

- 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| `grad` | 输入 | 前向输出对输入的梯度。 | 支持 1-8 维。 | `FLOAT`、`FLOAT16` | `ND` | `0-8` 维 | √ |
| `indices` | 输入 | 索引张量。 | 一维，元素值范围 `[0, numWeights)`。 | `INT32`、`INT64` | `ND` | `1` 维 | √ |
| `numWeights` | 输入 | 词表大小。 | 正整数。 | `INT64` | - | - | - |
| `paddingIdx` | 输入 | 填充索引。 | 范围 `[-1, numWeights)`，`-1` 表示不填充。 | `INT64` | - | - | - |
| `scaleGradByFreq` | 输入 | 是否按索引出现频次缩放梯度。 | `true`/`false`。 | `BOOL` | - | - | - |
| `y` | 输出 | 输出梯度张量。 | shape 为 `{numWeights, dimSize}`。 | 与 `grad` 相同 | `ND` | `0-8` 维 | - |
| `workspaceSize` | 输出 | 返回需要在 Device 侧申请的 workspace 大小。 | - | - | - | - | - |
| `executor` | 输出 | 返回算子执行器。 | 包含算子计算流程。 | - | - | - | - |

- 返回值

  `aclnnStatus`：返回状态码，具体参见 aclnn 返回码。

## aclnnEmbeddingDenseGrad

- 参数说明

| 参数名 | 输入/输出 | 描述 |
| ---- | ---- | ---- |
| `workspace` | 输入 | Device 侧申请的 workspace 内存地址。 |
| `workspaceSize` | 输入 | Device 侧申请的 workspace 大小，由 `aclnnEmbeddingDenseGradGetWorkspaceSize` 获取。 |
| `executor` | 输入 | 算子执行器，包含算子计算流程。 |
| `stream` | 输入 | 指定执行任务的 Stream。 |

- 返回值

  `aclnnStatus`：返回状态码，具体参见 aclnn 返回码。

## 约束说明

- `indices` 的元素值必须在 `[0, numWeights)` 范围内，`paddingIdx` 除外。
- `grad` 的第 0 维大小必须等于 `indices` 的元素个数。
- `y` 的 shape 为 `{numWeights, dimSize}`，其中 `dimSize` 等于 `grad` 的最后一维大小（即 `grad.shape[-1]`）。
- `scaleGradByFreq` 为 `true` 时，梯度会除以每个索引出现的次数。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| ACLNN 调用 | [test_aclnn_embedding_dense_grad.cpp](../examples/test_aclnn_embedding_dense_grad.cpp) | 通过 `aclnnEmbeddingDenseGradGetWorkspaceSize` 和 `aclnnEmbeddingDenseGrad` 两段式接口调用 EmbeddingDenseGrad 算子。 |
