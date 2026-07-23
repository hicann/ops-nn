# aclnnHardswishBackward

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A2 训练系列产品 | √ |
| Atlas A3 系列产品 | √ |

## 功能描述

`aclnnHardswishBackward` 提供 HardSwish 激活函数反向计算的两段式 ACLNN L2 接口。接口根据前向输入 `self` 所在区间计算局部梯度系数，并与上游梯度 `gradOutput` 相乘得到输出 `out`。

```text
out = 0                                  , self <= -3
out = gradOutput * (self / 3 + 0.5)      , -3 < self < 3
out = gradOutput                         , self >= 3
```

## 函数原型

```cpp
aclnnStatus aclnnHardswishBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnHardswishBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnHardswishBackwardGetWorkspaceSize 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | shape |
| :--- | :---: | :--- | :--- | :--- | :--- |
| gradOutput | 输入 | 上游梯度 Tensor | FLOAT16、FLOAT、BFLOAT16 | ND | 任意合法 shape |
| self | 输入 | HardSwish 前向输入 Tensor | 与 `gradOutput` 一致 | 与 `gradOutput` 一致 | 与 `gradOutput` 一致 |
| out | 输出 | HardSwish 反向梯度结果 Tensor | 与 `gradOutput` 一致 | 与 `gradOutput` 一致 | 与 `gradOutput` 一致 |
| workspaceSize | 输出 | 返回 device 侧 workspace 大小 | - | - | - |
| executor | 输出 | 返回执行器，供第二段接口使用 | - | - | - |

## aclnnHardswishBackward 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| workspace | 输入 | device 侧 workspace 地址；当 `workspaceSize` 为 0 时可传入空指针 |
| workspaceSize | 输入 | workspace 大小，由第一段接口返回 |
| executor | 输入 | 第一段接口返回的执行器 |
| stream | 输入 | 执行任务的 ACL stream |

## 返回值

| 返回值 | 说明 |
| :--- | :--- |
| ACLNN_SUCCESS | 执行成功 |
| 非 0 | 参数校验、资源申请或算子执行失败 |

## 约束与限制

- `gradOutput`、`self` 和 `out` 的数据类型必须一致。
- `gradOutput`、`self` 和 `out` 的 shape 必须一致，不支持广播。
- 支持 FLOAT16、FLOAT、BFLOAT16。
- 支持 ND。
- 支持非连续 Tensor。输入由接口转换为连续 Tensor，计算结果通过 ViewCopy 写入 `out`。
- 支持动态 shape、动态 rank、scalar Tensor 和空 Tensor。
- Tensor rank 不得超过框架支持的最大维度。

## 调用示例

参考 [test_aclnn_hard_swish_grad.cpp](../examples/test_aclnn_hard_swish_grad.cpp)。
