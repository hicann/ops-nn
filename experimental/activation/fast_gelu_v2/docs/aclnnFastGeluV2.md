# aclnnFastGeluV2

## 支持的产品型号

| 产品型号 | 芯片号 | 支持的数据类型 |
|---------|--------|---------------|
| Atlas A5 训练/推理系列 | Ascend950PR | float32, float16, bfloat16 |

## 功能描述

计算 FastGeluV2 激活函数。FastGeluV2 是 GELU 激活函数的一种高效近似实现变体，通过分段多项式近似替代标准 GELU 中的 erf/tanh 计算。

计算公式如下：

```text
FastGeluV2(x) = x * (sgn(x) * [-0.1444 * (clip(|0.7071 * x|, max=1.769) - 1.769)^2 + 0.5] + 0.5)
```

其中：

```text
sgn(x) = (x + 1e-12) / |x + 1e-12|
```

## aclnnFastGeluV2GetWorkspaceSize

### 函数原型

```cpp
aclnnStatus aclnnFastGeluV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

### 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|---------|------|
| x | 输入 | 公式中的输入 `x`，Device 侧的 aclTensor，数据类型支持 float32、float16、bfloat16。支持非连续 Tensor，数据格式支持 ND，最大支持 8 维。 |
| out | 输入 | 公式中的输出 `y`，Device 侧的 aclTensor，数据类型需要与 `x` 一致，shape 需要与 `x` 一致。支持非连续 Tensor，数据格式支持 ND。 |
| workspaceSize | 输出 | 返回用户需要在 Device 侧申请的 workspace 大小。 |
| executor | 输出 | 返回 op 执行器，包含了算子计算流程。 |

### 返回值

| 返回值 | 说明 |
|--------|------|
| ACLNN_SUCCESS (0) | 成功 |
| ACLNN_ERR_PARAM_NULLPTR (161001) | 必选参数为空指针 |
| ACLNN_ERR_PARAM_INVALID (161002) | 参数校验失败（dtype 不支持、shape 不匹配等） |
| 其他值 | 失败 |

## aclnnFastGeluV2

### 函数原型

```cpp
aclnnStatus aclnnFastGeluV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

### 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|---------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址。 |
| workspaceSize | 输入 | 在 Device 侧申请的 workspace 大小，由第一段接口 aclnnFastGeluV2GetWorkspaceSize 获取。 |
| executor | 输入 | op 执行器，包含了算子计算流程。 |
| stream | 输入 | 指定执行任务的 AscendCL Stream 流。 |

### 返回值

| 返回值 | 说明 |
|--------|------|
| ACLNN_SUCCESS (0) | 成功 |
| 其他值 | 失败 |

## 约束说明

1. 输入 `x` 和输出 `out` 的数据类型必须一致。
2. 输出 `out` 的 shape 必须与输入 `x` 的 shape 一致。
3. 输入张量维度不超过 8 维。
4. 不支持私有数据格式（Private Format）。
5. 输入 `x` 为空 Tensor 时，`workspaceSize` 返回 0，直接返回成功。

## 调用示例

以下代码摘自 [examples/test_aclnn_fast_gelu_v2.cpp](../examples/arch35/test_aclnn_fast_gelu_v2.cpp)，演示两段式 ACLNN 接口的完整调用流程。

### 1. 初始化 ACL 运行时

```cpp
#include "acl/acl.h"
#include "aclnn_fast_gelu_v2.h"

// 初始化 ACL 并设置设备
aclInit(nullptr);
int32_t deviceId = 0;
aclrtSetDevice(deviceId);
aclrtStream stream = nullptr;
aclrtCreateStream(&stream);
```

### 2. 准备输入输出张量

```cpp
// 定义 shape 和数据
std::vector<int64_t> shape = {2, 8};
int64_t totalElements = 16;
size_t dataBytes = totalElements * sizeof(float);

// 分配 Device 内存并拷贝输入数据
void* xDevAddr = nullptr;
aclrtMalloc(&xDevAddr, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(xDevAddr, dataBytes, hostInput.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);

// 计算 strides 并创建 aclTensor
auto strides = ComputeStrides(shape);  // row-major contiguous
aclTensor* xTensor = aclCreateTensor(
    shape.data(), shape.size(), ACL_FLOAT,
    strides.data(), 0, aclFormat::ACL_FORMAT_ND,
    shape.data(), shape.size(), xDevAddr);

// 同样创建输出 tensor（outDevAddr, outTensor）
```

### 3. Phase 1: GetWorkspaceSize

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnFastGeluV2GetWorkspaceSize(xTensor, outTensor, &workspaceSize, &executor);

// 按需分配 workspace
void* workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}
```

### 4. Phase 2: 执行算子

```cpp
aclnnFastGeluV2(workspace, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```

### 5. 获取结果并释放资源

```cpp
// D2H 拷贝结果
aclrtMemcpy(hostOutput.data(), dataBytes, outDevAddr, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);

// 释放资源
if (workspace) aclrtFree(workspace);
aclDestroyTensor(xTensor);
aclDestroyTensor(outTensor);
aclrtFree(xDevAddr);
aclrtFree(outDevAddr);
aclrtDestroyStream(stream);
aclrtResetDevice(deviceId);
aclFinalize();
```

### 编译运行

```bash
# 编译并运行示例
cd examples && bash run.sh --eager
```
