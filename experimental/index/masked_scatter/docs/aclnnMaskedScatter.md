# aclnnMaskedScatter

<!-- codespell:ignore inplace -->

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | √ |

## 功能说明

- 接口功能：根据掩码 `mask` 张量中元素为 `true` 的位置，将 `source` 中的元素按顺序写入 `selfRef` 对应位置。
- 当前算子提供的 ACLNN 接口为 in-place 形式，实际导出接口名为 `aclnnInplaceMaskedScatterGetWorkspaceSize` 和 `aclnnInplaceMaskedScatter`。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用
`aclnnInplaceMaskedScatterGetWorkspaceSize` 获取计算所需 workspace 大小以及执行器，再调用
`aclnnInplaceMaskedScatter` 执行计算。

```Cpp
aclnnStatus aclnnInplaceMaskedScatterGetWorkspaceSize(
    aclTensor* selfRef,
    const aclTensor* mask,
    const aclTensor* source,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

```Cpp
aclnnStatus aclnnInplaceMaskedScatter(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## aclnnInplaceMaskedScatterGetWorkspaceSize

- 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| `selfRef` | 输入 | 输入/输出 Tensor。 | 支持非连续 Tensor。 | `FLOAT`、`FLOAT16`、`DOUBLE`、`INT8`、`INT16`、`INT32`、`INT64`、`UINT8`、`BOOL`、`BFLOAT16` | `ND` | `0-8` 维 | √ |
| `mask` | 输入 | 掩码 Tensor。 | shape 维度不能大于 `selfRef`，且需与 `selfRef` 满足 [broadcast 关系](../../../../docs/zh/context/broadcast关系.md)。 | `BOOL`、`UINT8` | `ND` | `0-8` 维 | - |
| `source` | 输入 | 更新 Tensor。 | 数据类型需与 `selfRef` 相同，元素数量需大于等于 `mask` 中值为 `true` 的元素个数。 | 与 `selfRef` 相同 | `ND` | `0-8` 维 | - |
| `workspaceSize` | 输出 | 返回需要在 Device 侧申请的 workspace 大小。 | - | - | - | - | - |
| `executor` | 输出 | 返回算子执行器。 | 包含算子计算流程。 | - | - | - | - |

- 返回值

  `aclnnStatus`：返回状态码，具体参见 [aclnn 返回码](../../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

| 返回值 | 错误码 | 描述 |
| ---- | ---- | ---- |
| `ACLNN_ERR_PARAM_NULLPTR` | `161001` | 传入的 `selfRef`、`mask` 或 `source` 是空指针。 |
| `ACLNN_ERR_PARAM_INVALID` | `161002` | `selfRef` 或 `mask` 的数据类型不在支持范围内。 |
| `ACLNN_ERR_PARAM_INVALID` | `161002` | `source` 与 `selfRef` 的数据类型不同。 |
| `ACLNN_ERR_PARAM_INVALID` | `161002` | `mask` 与 `selfRef` 不满足 broadcast 关系，或 `mask` 维度大于 `selfRef`。 |

## aclnnInplaceMaskedScatter

- 参数说明

| 参数名 | 输入/输出 | 描述 |
| ---- | ---- | ---- |
| `workspace` | 输入 | Device 侧申请的 workspace 内存地址。 |
| `workspaceSize` | 输入 | Device 侧申请的 workspace 大小，由 `aclnnInplaceMaskedScatterGetWorkspaceSize` 获取。 |
| `executor` | 输入 | 算子执行器，包含算子计算流程。 |
| `stream` | 输入 | 指定执行任务的 Stream。 |

- 返回值

  `aclnnStatus`：返回状态码，具体参见 [aclnn 返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- `selfRef`、`mask`、`source` 的维度数均不超过 8。
- `mask` 的 shape 维度不能大于 `selfRef`，且需要和 `selfRef` 满足 broadcast 关系。
- `source` 的数据类型必须和 `selfRef` 相同。
- `source` 的元素数量需要大于等于 `mask` 中值为 `true` 的元素个数。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| ACLNN 调用 | [test_aclnn_masked_scatter.cpp](../examples/test_aclnn_masked_scatter.cpp) | 通过 `aclnnInplaceMaskedScatterGetWorkspaceSize` 和 `aclnnInplaceMaskedScatter` 两段式接口调用 MaskedScatter 算子。 |
