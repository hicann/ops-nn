# aclnnApplyAdamD

## 支持的产品型号

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>（ascend910b，DAV_2201） | √ |

## 功能说明

- **接口功能**：根据 Adam 算法对参数张量 `var` 做单步更新，并输出更新后的 `var`、`m`、`v`。
- **计算公式**：

  $$
  lr' = learning\_rate \times \frac{\sqrt{1 - beta2\_power}}{1 - beta1\_power}
  $$

  $$
  m_{new} = beta1 \times m + (1 - beta1) \times grad
  $$

  $$
  v_{new} = beta2 \times v + (1 - beta2) \times grad \times grad
  $$

  - `useNesterov = false`：

    $$
    var_{new} = var - lr' \times \frac{m_{new}}{epsilon + \sqrt{v_{new}}}
    $$

  - `useNesterov = true`：

    $$
    var_{new} = var - lr' \times \frac{m_{new} \times beta1 + (1 - beta1) \times grad}{epsilon + \sqrt{v_{new}}}
    $$

- **说明**：本接口沿用 `ApplyAdamD` 图模式原型，`beta1Power`、`beta2Power`、`lr`、`beta1`、`beta2`、`epsilon` 均为 shape size 为 1 的 `aclTensor`，而不是 `aclScalar`。`varOut`、`mOut`、`vOut` 可与 `var`、`m`、`v` 复用 Device 地址以实现原地更新。

## 接口原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用 `aclnnApplyAdamDGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含算子计算流程的执行器，再调用 `aclnnApplyAdamD` 接口执行计算。

```Cpp
aclnnStatus aclnnApplyAdamDGetWorkspaceSize(
    const aclTensor *var,
    const aclTensor *m,
    const aclTensor *v,
    const aclTensor *beta1Power,
    const aclTensor *beta2Power,
    const aclTensor *lr,
    const aclTensor *beta1,
    const aclTensor *beta2,
    const aclTensor *epsilon,
    const aclTensor *grad,
    bool useLocking,
    bool useNesterov,
    aclTensor *varOut,
    aclTensor *mOut,
    aclTensor *vOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnApplyAdamD(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnApplyAdamDGetWorkspaceSize

- **参数说明：**

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度/shape | 非连续 Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| var | 输入 | 待更新参数张量。 | FLOAT、FLOAT16、BFLOAT16 | ND | 0-8 | √ |
| m | 输入 | 一阶矩估计，与 `var` 同 shape、同 dtype。 | 与 `var` 一致 | ND | 与 `var` 一致 | √ |
| v | 输入 | 二阶矩估计，与 `var` 同 shape、同 dtype。 | 与 `var` 一致 | ND | 与 `var` 一致 | √ |
| beta1Power | 输入 | `beta1` 的幂次，标量 Tensor。 | 与 `var` 一致 | ND | shape size = 1 | √ |
| beta2Power | 输入 | `beta2` 的幂次，标量 Tensor。 | 与 `var` 一致 | ND | shape size = 1 | √ |
| lr | 输入 | 学习率，标量 Tensor。 | 与 `var` 一致 | ND | shape size = 1 | √ |
| beta1 | 输入 | 一阶矩衰减率，标量 Tensor。 | 与 `var` 一致 | ND | shape size = 1 | √ |
| beta2 | 输入 | 二阶矩衰减率，标量 Tensor。 | 与 `var` 一致 | ND | shape size = 1 | √ |
| epsilon | 输入 | 数值稳定常量，标量 Tensor。 | 与 `var` 一致 | ND | shape size = 1 | √ |
| grad | 输入 | 梯度张量，与 `var` 同 shape、同 dtype。 | 与 `var` 一致 | ND | 与 `var` 一致 | √ |
| useLocking | 输入 | 是否使用锁机制。端侧实现不改变数值计算。 | bool | - | - | - |
| useNesterov | 输入 | 是否使用 Nesterov 分支。 | bool | - | - | - |
| varOut | 输出 | 更新后的 `var`，shape/dtype 与 `var` 一致。 | 与 `var` 一致 | ND | 与 `var` 一致 | √ |
| mOut | 输出 | 更新后的 `m`，shape/dtype 与 `var` 一致。 | 与 `var` 一致 | ND | 与 `var` 一致 | √ |
| vOut | 输出 | 更新后的 `v`，shape/dtype 与 `var` 一致。 | 与 `var` 一致 | ND | 与 `var` 一致 | √ |
| workspaceSize | 输出 | 返回需要在 Device 侧申请的 workspace 大小。 | uint64_t* | - | - | - |
| executor | 输出 | 返回 op 执行器，包含算子计算流程。 | aclOpExecutor** | - | - | - |

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见[aclnn 返回码](../../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

| 返回码 | 错误码 | 描述 |
| :--- | :--- | :--- |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入的计算输入/输出参数、`workspaceSize` 或 `executor` 是空指针。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 输入 dtype 不在 {FLOAT、FLOAT16、BFLOAT16} 范围内。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 输入/输出 dtype 与 `var` 不一致。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | `m`、`v`、`grad` 或输出 shape 与 `var` 不一致。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | `beta1Power`、`beta2Power`、`lr`、`beta1`、`beta2`、`epsilon` 的 shape size 不为 1。 |
| ACLNN_ERR_INNER_CREATE_EXECUTOR | 361001 | 创建执行器失败。 |
| ACLNN_ERR_INNER_NULLPTR | 361002 | 内部 contiguous、L0 调用或 ViewCopy 失败。 |

## aclnnApplyAdamD

- **参数说明：**

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| workspace | 输入 | 在 Device 侧申请的 workspace 地址。 |
| workspaceSize | 输入 | workspace 大小，由第一段接口 `aclnnApplyAdamDGetWorkspaceSize` 获取。 |
| executor | 输入 | op 执行器，包含算子计算流程。 |
| stream | 输入 | 指定执行任务的 AscendCL Stream。 |

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见[aclnn 返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束与说明

- `var`、`m`、`v`、`grad` 必须同 shape、同 dtype。
- `beta1Power`、`beta2Power`、`lr`、`beta1`、`beta2`、`epsilon` 必须为 shape size = 1 的 Tensor，且 dtype 与 `var` 一致。
- `varOut`、`mOut`、`vOut` 的 shape/dtype 必须与 `var` 一致；需要原地语义时，输出 Tensor 可复用对应输入 Tensor 的 Device 地址。
- `1 - beta1Power` 作分母须非 0，由调用方保证。
- 本算子为逐元素计算，无跨元素归约/原子累加，默认确定性实现。
- 本算子原型已废弃，建议新业务优先使用 `ApplyAdam`；本接口用于 `ApplyAdamD` 兼容调用。

## 调用示例

完整示例见 [test_aclnn_apply_adam_d.cpp](../examples/test_aclnn_apply_adam_d.cpp)。核心调用流程如下：

```Cpp
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
bool useLocking = false;
bool useNesterov = false;

ret = aclnnApplyAdamDGetWorkspaceSize(var, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad,
    useLocking, useNesterov, varOut, mOut, vOut, &workspaceSize, &executor);
CHECK_RET(ret == ACL_SUCCESS, return ret);

void *workspaceAddr = nullptr;
if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
}

ret = aclnnApplyAdamD(workspaceAddr, workspaceSize, executor, stream);
CHECK_RET(ret == ACL_SUCCESS, return ret);
```
