# aclnnSoftMarginLoss

## 支持的产品型号

| 产品系列 | 产品型号 |
|---------|---------|
| Atlas A2 训练系列产品 | Atlas 800T A2、Atlas 800I A2、Atlas 900 A2 PoD、Atlas 200I A2 |

## 功能描述

计算输入张量 `self` 与目标张量 `target` 之间的 SoftMarginLoss，即逐元素计算：

$$\text{loss}_i = \log\!\left(1 + e^{-\text{target}_i \times \text{self}_i}\right)$$

并根据 `reduction` 参数对结果进行归约：

- `reduction=0`（none）：不归约，输出 shape 与输入相同。
- `reduction=1`（mean）：对所有元素求均值，输出为标量。
- `reduction=2`（sum）：对所有元素求和，输出为标量。

- `self` 与 `target` 须具有相同的 shape 和数据类型。
- 支持数据类型：float32、float16。

## 函数原型

```cpp
aclnnStatus aclnnSoftMarginLossGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *target,
    int64_t reduction,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnSoftMarginLoss(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnSoftMarginLossGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|-------|---------|------|
| self | 输入 | 预测值张量。数据类型：float32、float16。数据格式：ND。支持非连续 tensor。 |
| target | 输入 | 目标值张量（通常为 +1 或 -1）。数据类型须与 self 相同。数据格式：ND。shape 须与 self 相同。 |
| reduction | 输入 | 归约模式。int64_t 类型，取值：0（none）、1（mean，默认）、2（sum）。 |
| out | 输出 | 输出张量。数据类型与 self 相同。reduction=0 时 shape 与 self 相同；reduction=1 或 2 时为标量（0 维 tensor）。 |
| workspaceSize | 输出 | 算子执行所需 workspace 大小，单位为 Byte。由本函数返回，调用方须据此分配 workspace 内存。 |
| executor | 输出 | 算子执行器，包含算子计算流信息，由本函数返回后传入 aclnnSoftMarginLoss 执行。 |

### 返回值说明

返回 `aclnnStatus` 错误码，详见 [aclnn 错误码](#错误码)。

## aclnnSoftMarginLoss

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|-------|---------|------|
| workspace | 输入 | workspace 内存地址。若 workspaceSize 为 0，可传入 nullptr。 |
| workspaceSize | 输入 | workspace 大小，由 aclnnSoftMarginLossGetWorkspaceSize 返回。 |
| executor | 输入 | 算子执行器，由 aclnnSoftMarginLossGetWorkspaceSize 返回。 |
| stream | 输入 | ACL stream，用于异步调度算子执行。 |

### 返回值说明

返回 `aclnnStatus` 错误码，详见 [aclnn 错误码](#错误码)。

## 错误码

| 错误码 | 描述 |
|-------|------|
| ACLNN_SUCCESS（0） | 执行成功。 |
| ACLNN_ERR_PARAM_NULLPTR（161001） | 输入/输出 tensor 指针为空。 |
| ACLNN_ERR_PARAM_INVALID（161002） | 参数非法，包括：数据类型不支持、self 与 target 数据类型不一致、reduction 值不在 {0,1,2} 范围内、out shape 与预期不一致等。 |
| ACLNN_ERR_INNER_CREATE_EXECUTOR | 内部创建算子执行器失败。 |
| ACLNN_ERR_INNER_NULLPTR | 内部 tensor 分配失败。 |
| ACLNN_ERR_INNER_INFERSHAPE_ERROR | 内部 InferShape 失败。 |

## 约束说明

- `self` 与 `target` 须为相同数据类型和相同 shape。
- `reduction` 仅支持 0（none）、1（mean）、2（sum），其他值返回 `ACLNN_ERR_PARAM_INVALID`。
- `out` 的 shape 须与 reduction 模式匹配：reduction=0 时与 self 相同，reduction=1 或 2 时为 0 维标量（**注意：不是 shape=[1]，而是 dimNum=0 的标量 tensor**）。
- 支持空 tensor（元素数为 0），此时 workspaceSize 为 0，直接返回成功。
- workspace 须在调用 `aclnnSoftMarginLoss` 之前分配，在 stream 中算子执行完成后方可释放。

## 调用示例

以下示例展示了 SoftMarginLoss 算子的完整调用流程（reduction=mean）：

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_soft_margin_loss.h"

int main() {
    // 1. 初始化 ACL 及设备
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // 2. 准备输入数据（fp32，shape=[4, 8]）
    int64_t shape[] = {4, 8};
    int64_t strides[] = {8, 1};

    float self_host[] = {
         0.5f,  1.0f, -0.3f,  2.0f,  0.1f, -1.5f,  0.8f, -0.2f,
        -1.0f,  0.7f,  1.5f, -0.5f,  0.3f,  0.9f, -0.8f,  1.2f,
         0.4f, -0.6f,  1.1f, -1.3f,  0.6f, -0.4f,  1.4f,  0.2f,
        -0.9f,  1.3f, -0.1f,  0.0f,  1.6f, -1.1f,  0.5f, -0.7f
    };
    float target_host[] = {
         1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
         1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f
    };
    size_t nbytes = 32 * sizeof(float);

    void *self_dev = nullptr, *target_dev = nullptr, *out_dev = nullptr;
    aclrtMalloc(&self_dev,   nbytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&target_dev, nbytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&out_dev,    sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(self_dev,   nbytes, self_host,   nbytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(target_dev, nbytes, target_host, nbytes, ACL_MEMCPY_HOST_TO_DEVICE);

    // 3. 创建 aclTensor
    aclTensor *self_t   = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                          ACL_FORMAT_ND, shape, 2, self_dev);
    aclTensor *target_t = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                                          ACL_FORMAT_ND, shape, 2, target_dev);

    // reduction=mean/sum 时 out 为 0 维标量 tensor
    aclTensor *out_t = aclCreateTensor(nullptr, 0, ACL_FLOAT, nullptr, 0,
                                       ACL_FORMAT_ND, nullptr, 0, out_dev);

    // 4. 查询 workspace 大小并分配
    int64_t reduction = 1;  // mean
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnSoftMarginLossGetWorkspaceSize(self_t, target_t, reduction, out_t,
                                        &workspaceSize, &executor);

    void *workspace = nullptr;
    if (workspaceSize > 0)
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);

    // 5. 执行算子
    aclnnSoftMarginLoss(workspace, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);

    // 6. 取回结果
    float out_host = 0.0f;
    aclrtMemcpy(&out_host, sizeof(float), out_dev, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    printf("SoftMarginLoss (mean) = %f\n", out_host);
    // 期望: 0.781925

    // 7. 释放资源
    if (workspace) aclrtFree(workspace);
    aclrtFree(self_dev); aclrtFree(target_dev); aclrtFree(out_dev);
    aclDestroyTensor(self_t); aclDestroyTensor(target_t); aclDestroyTensor(out_t);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```

> **reduction=none 示例**：若 `reduction=0`，则 `out` 的 shape 须与 `self` 相同（例如 `[4, 8]`），输出为逐元素的 loss 值。
