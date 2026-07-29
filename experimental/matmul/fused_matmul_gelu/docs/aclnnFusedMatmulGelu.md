# aclnnFusedMatmulGelu

## 支持的产品型号

| 产品                              | 是否支持 |
| :------------------------------ | :--: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |   √  |

## 接口原型

```cpp
aclnnStatus aclnnFusedMatmulGeluGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* bias,
    int64_t approximate,
    aclTensor* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnFusedMatmulGelu(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## 功能说明

`aclnnFusedMatmulGelu` 接口用于调用自定义 `FusedMatmulGelu` 融合算子，实现 MatMul/Linear、可选偏置加法和 GELU 激活函数的融合计算。

计算公式如下：

```text
y = GELU(x * weight^T + bias)
```

当 `bias` 为空时，计算公式为：

```text
y = GELU(x * weight^T)
```

## 参数说明

| 参数名           | 输入/输出 | 说明                                                        |
| ------------- | ----- | --------------------------------------------------------- |
| x             | 输入    | 输入张量，shape 为 `[..., K]`，数据类型支持 FLOAT16、BFLOAT16，数据格式为 ND。 |
| weight        | 输入    | 权重张量，shape 为 `[N, K]`，数据类型需要与 `x` 保持一致，数据格式为 ND。          |
| bias          | 输入    | 可选偏置张量，shape 为 `[N]`，数据类型需要与 `x` 保持一致，数据格式为 ND。           |
| y             | 输出    | 输出张量，shape 为 `[..., N]`，数据类型需要与 `x` 保持一致，数据格式为 ND。        |
| workspaceSize | 输出    | 返回执行该算子所需的 workspace 大小。                                  |
| executor      | 输出    | 返回算子执行器。                                                  |
| workspace     | 输入    | 算子执行所需的 workspace 地址。                                     |
| stream        | 输入    | ACL runtime stream。                                       |

## 约束说明

* `x`、`weight`、`bias` 和 `y` 的数据类型需要保持一致。
* 当前支持 FLOAT16、BFLOAT16 数据类型。
* 当前支持 ND 数据格式。
* `x` 的最后一维大小需要与 `weight` 的最后一维大小保持一致。
* `bias` 为可选输入；当 `bias` 不为空时，其 shape 需要为 `[N]`。
* `approximate` 当前仅支持取值为1（tanh近似模式）。

## 调用说明

调用样例请参考：

[test_aclnn_fused_matmul_gelu.cpp](../examples/test_aclnn_fused_matmul_gelu.cpp)
