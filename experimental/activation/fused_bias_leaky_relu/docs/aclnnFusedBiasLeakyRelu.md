# aclnnFusedBiasLeakyRelu

## 支持的产品型号

| 产品系列 | 芯片型号 | 支持状态 |
|---------|---------|---------|
| Atlas A2 训练系列 | Ascend910B | YES |

## 功能描述

计算融合的 Bias Add + LeakyReLU 激活函数。对输入张量 x 加上偏置 bias 后，应用带缩放因子的 LeakyReLU 激活。

计算公式：

```text
当 x + bias >= 0 时：y = (x + bias) * scale
当 x + bias <  0 时：y = (x + bias) * negative_slope * scale
```

其中：

- x 为输入数据张量
- bias 为偏置张量，shape 与 x 完全相同
- negative_slope 为 LeakyReLU 负半轴斜率（float 类型属性）
- scale 为缩放因子（float 类型属性）
- y 为输出张量，shape 和 dtype 与 x 相同

## 函数原型

```cpp
aclnnStatus aclnnFusedBiasLeakyReluGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *bias,
    double negativeSlope,
    double scale,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnFusedBiasLeakyRelu(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 参数说明

### aclnnFusedBiasLeakyReluGetWorkspaceSize

| 参数名 | 输入/输出 | 类型 | 说明 |
|--------|---------|------|------|
| x | 输入 | const aclTensor* | 输入数据张量。数据类型支持 FLOAT16、FLOAT。数据格式支持 ND。最高支持 8 维。 |
| bias | 输入 | const aclTensor* | 偏置张量。数据类型和 shape 必须与 x 一致。数据格式支持 ND。 |
| negativeSlope | 输入 | double | LeakyReLU 负半轴斜率系数。可选，默认值 0.2。 |
| scale | 输入 | double | 缩放因子。可选，默认值 1.414213562373（即 √2）。 |
| out | 输入 | const aclTensor* | 输出张量（需预分配）。数据类型和 shape 必须与 x 一致。数据格式支持 ND。 |
| workspaceSize | 输出 | uint64_t* | 返回执行算子所需的 workspace 大小（字节）。 |
| executor | 输出 | aclOpExecutor** | 返回算子执行器。 |

### aclnnFusedBiasLeakyRelu

| 参数名 | 输入/输出 | 类型 | 说明 |
|--------|---------|------|------|
| workspace | 输入 | void* | 在 Device 侧申请的 workspace 内存地址。 |
| workspaceSize | 输入 | uint64_t | workspace 内存大小（字节），由第一段接口 aclnnFusedBiasLeakyReluGetWorkspaceSize 获取。 |
| executor | 输入 | aclOpExecutor* | 算子执行器，由第一段接口 aclnnFusedBiasLeakyReluGetWorkspaceSize 获取。 |
| stream | 输入 | aclrtStream | ACL stream 流。 |

## 返回值

| 返回值 | 说明 |
|--------|------|
| ACLNN_SUCCESS (0) | 成功 |
| ACLNN_ERR_PARAM_NULLPTR | 输入参数中存在空指针 |
| ACLNN_ERR_PARAM_INVALID | 输入参数不合法（dtype 不匹配、shape 不一致、不支持的 dtype、不支持的 format、维度超限等） |
| ACLNN_ERR_INNER_CREATE_EXECUTOR | 内部创建执行器失败 |
| ACLNN_ERR_INNER_NULLPTR | 内部操作返回空指针 |

## 约束说明

- x、bias、out 三者的数据类型（dtype）必须一致，仅支持 FLOAT16 和 FLOAT。
- x 和 bias 的 shape 必须完全相同，不支持广播。
- out 的 shape 必须与 x 相同。
- negativeSlope 和 scale 为 double 类型可选属性，带默认值（negative_slope=0.2，scale=1.414213562373）。
- 支持的数据格式为 ND，不支持 Private format。
- 张量最高支持 8 维。
- 当输入张量为空张量（包含 0 元素）时，workspaceSize 返回 0，直接返回成功。

## 调用示例

请参考 [examples/test_aclnn_fused_bias_leaky_relu.cpp](../examples/test_aclnn_fused_bias_leaky_relu.cpp)。

该示例演示了完整的 ACLNN 两段式调用流程：

1. 初始化 ACL 环境（aclInit、aclrtSetDevice、aclrtCreateStream）
2. 准备输入数据并创建 aclTensor
3. 调用 aclnnFusedBiasLeakyReluGetWorkspaceSize 获取 workspace 大小和 executor
4. 分配 workspace 内存
5. 调用 aclnnFusedBiasLeakyRelu 执行计算
6. 同步流并拷贝结果回 Host
7. 清理资源
