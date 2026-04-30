# aclnnSmoothL1LossV2

## 产品支持情况

| 产品 | 是否支持 |
|:--|:--:|
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- 接口功能: 计算 SmoothL1 损失函数，支持 reduction 模式 `none`、`mean`、`sum`。
- 计算公式:

  当 `reduction` 为 `none` 时:

  $$
  \ell(predict, label) = L = \{l_1, \dots, l_N\}^\top
  $$

  其中:

  $$
  l_n =
  \begin{cases}
  0.5(predict_n - label_n)^2/\sigma, & |predict_n - label_n| < \sigma \\
  |predict_n - label_n| - 0.5\sigma, & \text{otherwise}
  \end{cases}
  $$

  当 `reduction` 为 `mean` 或 `sum` 时:

  $$
  \ell(predict, label)=
  \begin{cases}
  mean(L), & \text{if reduction} = \text{mean} \\
  sum(L), & \text{if reduction} = \text{sum}
  \end{cases}
  $$

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnSmoothL1LossGetWorkspaceSize` 获取 workspace 大小和执行器，再调用 `aclnnSmoothL1Loss` 执行计算。

```cpp
aclnnStatus aclnnSmoothL1LossGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* target,
    int64_t reduction,
    float beta,
    aclTensor* result,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnSmoothL1Loss(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
```

## aclnnSmoothL1LossGetWorkspaceSize

- 参数说明:

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|:--|:--:|:--|:--|:--|:--|:--|:--:|
| self(aclTensor*) | 输入 | 输入张量，公式中的 `predict`。 | shape 需要与 `target` 完全一致，不支持 broadcast。dtype 需满足与 `target` 的类型推导规则。 | FLOAT、FLOAT16、BFLOAT16 | ND、NCL、NCHW、NHWC | 1-8 | √ |
| target(aclTensor*) | 输入 | 标签张量，公式中的 `label`。 | shape 需要与 `self` 完全一致，不支持 broadcast。dtype 需满足与 `self` 的类型推导规则。 | FLOAT、FLOAT16、BFLOAT16 | ND、NCL、NCHW、NHWC | 1-8 | √ |
| reduction(int64_t) | 输入 | 输出归约方式。 | 支持 `0(none)\|1(mean)\|2(sum)`。`none` 不归约，`mean` 输出均值，`sum` 输出求和。 | INT64 | - | - | - |
| beta(float) | 输入 | 平滑阈值，等价于算子属性 `sigma`。 | 建议传入正数；当前实现中非正数会按默认值处理。 | FLOAT | - | - | - |
| result(aclTensor*) | 输出 | 输出损失值。 | 当 `reduction=none` 时，shape 与 `self`/`target` 一致; 当 `reduction=mean` 或 `sum` 时，输出为标量。dtype 为可从 `self`/`target` 推导并可转换的类型。 | FLOAT、FLOAT16、BFLOAT16 | ND、NCL、NCHW、NHWC | 0-8 | √ |
| workspaceSize(uint64_t*) | 输出 | Device 侧申请的 workspace 大小。 | - | - | - | - | - |
| executor(aclOpExecutor**) | 输出 | 执行器，包含算子执行流程。 | - | - | - | - | - |

- 返回值:

  `aclnnStatus`: 返回状态码。

  第一段接口完成参数检查，以下场景会返回错误:

<table class="tg" style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 269px">
  <col style="width: 135px">
  <col style="width: 746px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">返回值</th>
      <th class="tg-0pky">错误码</th>
      <th class="tg-0pky">描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">`self`、`target` 或 `result` 为空指针。</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="4">161002</td>
      <td class="tg-0pky">`self`、`target` 或 `result` 的 dtype 不在支持范围内。</td>
    </tr>
    <tr>
      <td class="tg-0pky">`self` 与 `target` 的 shape 不一致。</td>
    </tr>
    <tr>
      <td class="tg-0pky">`reduction` 不在 `0/1/2` 范围内。</td>
    </tr>
    <tr>
      <td class="tg-0pky">`reduction=0` 时，`result` shape 与输入 shape 不一致。</td>
    </tr>
  </tbody>
</table>

## aclnnSmoothL1Loss

- 参数说明:

| 参数名 | 输入/输出 | 描述 |
|:--|:--:|:--|
| workspace | 输入 | Device 侧申请的 workspace 地址。 |
| workspaceSize | 输入 | Device 侧申请的 workspace 大小，由第一段接口返回。 |
| executor | 输入 | 执行器，包含算子执行流程。 |
| stream | 输入 | 指定执行任务的 stream。 |

- 返回值:

  `aclnnStatus`: 返回状态码。

## 约束说明

- `self` 与 `target` 不支持 broadcast，shape 必须严格一致。
- `self` 与 `target` 的 dtype 必须可互推导并支持输出转换。
- 建议 `beta` 传入正数；当前实现中非正数会按默认值处理。
- `reduction=none` 输出逐元素; `reduction=mean/sum` 输出标量。

## 调用示例

示例代码如下：

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_smooth_l1_loss.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
#if !HAS_ACLNN_SMOOTH_L1_LOSS_V2
    LOG_PRINT("aclnn_smooth_l1_loss.h not found, skip example.\n");
    return 0;
#endif

    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> selfShape = {2, 2};
    std::vector<int64_t> targetShape = {2, 2};
    std::vector<int64_t> resultShape = selfShape;

    std::vector<float> selfData = {0, 1, 2, 3};
    std::vector<float> targetData = {0, 0.5, 2, 1};
    std::vector<float> resultData = {0, 0, 0, 0};

    void* selfDeviceAddr = nullptr;
    void* targetDeviceAddr = nullptr;
    void* resultDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* target = nullptr;
    aclTensor* result = nullptr;

    ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(targetData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(resultData, resultShape, &resultDeviceAddr, aclDataType::ACL_FLOAT, &result);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t reduction = 0; // 0:none, 1:mean, 2:sum
    float beta = 1.0f;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnSmoothL1LossGetWorkspaceSize(self, target, reduction, beta, result, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSmoothL1LossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnSmoothL1Loss(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSmoothL1Loss failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(resultShape);
    std::vector<float> resultOut(size, 0);
    ret = aclrtMemcpy(
        resultOut.data(), resultOut.size() * sizeof(resultOut[0]), resultDeviceAddr, size * sizeof(resultOut[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultOut[i]);
    }

    aclDestroyTensor(self);
    aclDestroyTensor(target);
    aclDestroyTensor(result);

    aclrtFree(selfDeviceAddr);
    aclrtFree(targetDeviceAddr);
    aclrtFree(resultDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
