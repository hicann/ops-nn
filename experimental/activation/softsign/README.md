# Softsign

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR/Ascend 950DT | 是 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | 是 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | 是 |

## 功能说明

完成 Softsign 激活函数计算。Softsign 是一种类似于 tanh 的激活函数，但其收敛速度为多项式级别而非指数级别。对标 PyTorch `torch.nn.Softsign` / `torch.nn.functional.softsign`。

计算公式：

$$
y = \frac{x}{1 + |x|}
$$

其中：
- $x$ 为输入张量的元素。
- $y$ 为输出张量的元素。
- 输出值域为 $(-1, 1)$。

## 调用方式

| 调用方式 | 是否支持 |
| :------- | :------: |
| ACLNN 调用 | 是 |

## ACLNN 接口

### 函数原型

每个算子分为两段式接口，必须先调用 `aclnnSoftsignGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 `aclnnSoftsign` 接口执行计算。

```cpp
aclnnStatus aclnnSoftsignGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnSoftsign(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

### 参数说明

**aclnnSoftsignGetWorkspaceSize：**

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度 |
| :----- | :-------: | :--- | :------- | :--- |
| self | 输入 | 输入张量，支持空 Tensor 和非连续 Tensor。 | FLOAT、FLOAT16、BFLOAT16 | 0-8 维 |
| out | 输出 | 输出张量，数据类型和 shape 需要与 self 一致。 | FLOAT、FLOAT16、BFLOAT16 | 0-8 维 |
| workspaceSize | 输出 | 返回需要在 Device 侧申请的 workspace 大小。 | uint64_t* | - |
| executor | 输出 | 返回 op 执行器，包含了算子计算流程。 | aclOpExecutor** | - |

**aclnnSoftsign：**

| 参数名 | 输入/输出 | 描述 |
| :----- | :-------: | :--- |
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址。 |
| workspaceSize | 输入 | 在 Device 侧申请的 workspace 大小，由第一段接口获取。 |
| executor | 输入 | op 执行器，包含了算子计算流程。 |
| stream | 输入 | 指定执行任务的 Stream。 |

### 返回值

aclnnStatus 返回状态码。

| 返回值 | 错误码 | 描述 |
| :----- | :----: | :--- |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | self 或 out 存在空指针。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | self 的数据类型不在支持的范围之内。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | out 的数据类型与 self 不一致。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | out 的 shape 与 self 不一致。 |

## 约束说明

- self 仅支持 FLOAT、FLOAT16、BFLOAT16 数据类型。
- out 的数据类型必须与 self 一致。
- out 的 shape 必须与 self 的 shape 完全一致，无广播行为。
- self 和 out 支持 0-8 维。
- self 支持空 Tensor（元素个数为 0），此时 out 也为空 Tensor，不执行任何计算。
- self 和 out 均支持非连续 Tensor。
- Softsign 默认确定性实现。

## 调用示例

### ACLNN 调用示例

通过 aclnn 两段式接口调用 Softsign 算子的完整示例，请参考 [examples/test_aclnn_softsign.cpp](examples/test_aclnn_softsign.cpp)。

主要流程：
1. 初始化 aclnn 环境、设置设备、创建 Stream
2. 创建输入/输出 Tensor，分配 Device 内存
3. 将输入数据从 Host 拷贝到 Device
4. 调用 `aclnnSoftsignGetWorkspaceSize` 获取 workspace 大小和执行器
5. 分配 workspace 内存，调用 `aclnnSoftsign` 执行计算
6. 同步 Stream，将结果从 Device 拷贝回 Host
7. 对比结果与 CPU 参考值
8. 释放资源
