# aclnnEluBackward

## 产品支持情况

| 产品 | 是否支持 |
|:--|:--:|
| <term>Atlas A2 训练系列产品 / Atlas A2 推理系列产品</term> | 支持 |

## 功能说明

- 接口功能：完成 ELU 激活函数的反向计算。
- 输入说明：
  - `gradOutput`：上游梯度。
  - `selfOrResult`：当 `isResult=false` 时表示 ELU 前向输入；当 `isResult=true` 时表示 ELU 前向输出。

- 计算公式：

当 `selfOrResult > 0` 时：

$$
gradInput = gradOutput \times scale
$$

当 `selfOrResult ≤ 0` 且 `isResult = false` 时：

$$
gradInput = gradOutput \times \alpha \times scale \times inputScale \times \exp(selfOrResult \times inputScale)
$$

当 `selfOrResult ≤ 0` 且 `isResult = true` 时：

$$
gradInput = gradOutput \times inputScale \times (selfOrResult + \alpha \times scale)
$$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)。必须先调用 `aclnnEluBackwardGetWorkspaceSize` 获取所需 workspace 大小和执行器，再调用 `aclnnEluBackward` 执行计算。

```Cpp
aclnnStatus aclnnEluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput,
    const aclScalar* alpha,
    const aclScalar* scale,
    const aclScalar* inputScale,
    bool isResult,
    const aclTensor* selfOrResult,
    aclTensor* gradInput,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

```Cpp
aclnnStatus aclnnEluBackward(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## aclnnEluBackwardGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | shape 约束 | 非连续 Tensor |
|:--|:--:|:--|:--|:--|:--|:--:|
| `gradOutput` | 输入 | ELU 反向的上游梯度 | `FLOAT`、`FLOAT16`、`BFLOAT16` | 推荐 `ND` | 与 `selfOrResult`、`gradInput` shape 完全一致，维度数不超过 8 | 支持 |
| `alpha` | 输入 | ELU 激活系数 | 可转换为 `FLOAT` 的 `aclScalar` | - | 标量 | - |
| `scale` | 输入 | 输出缩放系数 | 可转换为 `FLOAT` 的 `aclScalar` | - | 标量 | - |
| `inputScale` | 输入 | 输入缩放系数 | 可转换为 `FLOAT` 的 `aclScalar` | - | 标量 | - |
| `isResult` | 输入 | `false` 表示 `selfOrResult` 为前向输入，`true` 表示 `selfOrResult` 为前向输出 | `bool` | - | 标量 | - |
| `selfOrResult` | 输入 | ELU 前向输入或前向输出 | `FLOAT`、`FLOAT16`、`BFLOAT16`，且必须与 `gradOutput` dtype 一致 | 推荐 `ND` | 与 `gradOutput`、`gradInput` shape 完全一致，维度数不超过 8 | 支持 |
| `gradInput` | 输出 | 反向计算输出，即对前向输入的梯度 | 需为 `gradOutput` 可转换到的类型，通常与 `gradOutput` 保持一致 | 推荐 `ND` | 与 `gradOutput`、`selfOrResult` shape 完全一致，维度数不超过 8 | 支持 |
| `workspaceSize` | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | `uint64_t*` | - | - | - |
| `executor` | 输出 | 返回算子执行器，包含计算流程 | `aclOpExecutor**` | - | - | - |

### 参数约束与补充说明

- `gradOutput`、`selfOrResult` 的 dtype 必须一致。
- `gradOutput`、`selfOrResult`、`gradInput` 的 shape 必须一致，不支持 broadcast。
- 当 `isResult=true` 时，`alpha` 不能小于 0。

### 返回值说明

`aclnnStatus`：返回状态码，具体可参考 [aclnn 返回码](../../../../docs/zh/context/aclnn返回码.md)。

第一段接口会完成参数校验，典型错误场景如下：

| 返回码 | 错误码 | 说明 |
|:--|:--:|:--|
| `ACLNN_ERR_PARAM_NULLPTR` | `161001` | `gradOutput`、`alpha`、`scale`、`inputScale`、`selfOrResult`、`gradInput`、`workspaceSize` 或 `executor` 为空指针 |
| `ACLNN_ERR_PARAM_INVALID` | `161002` | `gradOutput` / `selfOrResult` dtype 不受支持 |
| `ACLNN_ERR_PARAM_INVALID` | `161002` | `gradOutput` 与 `selfOrResult` dtype 不一致 |
| `ACLNN_ERR_PARAM_INVALID` | `161002` | `gradOutput`、`selfOrResult`、`gradInput` shape 不一致 |
| `ACLNN_ERR_PARAM_INVALID` | `161002` | 输入维度数超过 8 |
| `ACLNN_ERR_PARAM_INVALID` | `161002` | `alpha`、`scale`、`inputScale` 不能转换为 `FLOAT` |
| `ACLNN_ERR_PARAM_INVALID` | `161002` | `isResult=true` 且 `alpha<0` |

## aclnnEluBackward

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|:--|:--:|:--|
| `workspace` | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| `workspaceSize` | 输入 | Device 侧申请的 workspace 大小，由 `aclnnEluBackwardGetWorkspaceSize` 返回 |
| `executor` | 输入 | 算子执行器，包含完整计算流程 |
| `stream` | 输入 | 指定执行任务的 `aclrtStream` |

### 返回值说明

`aclnnStatus`：返回状态码，具体可参考 [aclnn 返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 调用示例

示例代码参考 [examples/test_aclnn_elu_grad_v2.cpp](../examples/test_aclnn_elu_grad_v2.cpp)，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。
