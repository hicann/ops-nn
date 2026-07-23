# aclnnSeluBackward

## 产品支持情况

| 产品 | 是否支持 |
|:--|:--:|
| Atlas A2 训练系列产品 | √ |
| Atlas A3 系列产品 | √ |
| 其他产品 | × |

## 功能说明

`aclnnSeluBackward` 计算 SELU 激活函数的反向梯度：

$$
\text{gradInput} =
\begin{cases}
\text{gradOutput} \times \text{scale}, & \text{result} \ge 0 \\
\text{gradOutput} \times (\text{result} + \text{scale} \times \alpha), & \text{result} < 0
\end{cases}
$$

其中 `result` 为 SELU 前向输出，`gradOutput` 为上游梯度。

## 函数原型

先调用第一段接口获取 workspace 大小和执行器，再调用第二段接口执行。

```cpp
aclnnStatus aclnnSeluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput,
    const aclTensor* result,
    aclTensor* gradInput,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

```cpp
aclnnStatus aclnnSeluBackward(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|:--|:--:|:--|
| gradOutput | 输入 | 上游梯度，支持 FLOAT、FLOAT16、BFLOAT16、INT32、INT8、UINT8 |
| result | 输入 | SELU 前向输出，数据类型和 Shape 必须与 gradOutput 一致 |
| gradInput | 输出 | 输入梯度，数据类型和 Shape 必须与 gradOutput 一致 |
| workspaceSize | 输出 | Device workspace 大小 |
| executor | 输出 | 包含计算流程的执行器 |
| workspace | 输入 | Device workspace 地址；workspaceSize 为 0 时可传空指针 |
| stream | 输入 | 执行计算的 ACL Stream |

三个 Tensor 均使用 ND 格式。接口支持空 Tensor 和非连续 Tensor；非连续输入会在执行器中转换为连续 Tensor。

## 返回值

| 返回值 | 说明 |
|:--|:--|
| ACLNN_SUCCESS | 接口执行成功 |
| ACLNN_ERR_PARAM_NULLPTR | 输入、输出、workspaceSize 或 executor 等必选参数为空 |
| ACLNN_ERR_PARAM_INVALID | 数据类型不支持、输入输出数据类型不一致或 Shape 不一致 |

## 编译与运行

在仓库根目录完成 experimental 算子打包并安装生成的自定义算子包，然后执行：

```bash
bash build.sh --run_example selu_grad eager cust --vendor_name=custom --soc=ascend910b --experimental
```

样例源码见 [test_aclnn_selu_grad.cpp](../examples/test_aclnn_selu_grad.cpp)。
