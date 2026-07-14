# aclnnGroupNormalizationGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | √ |

## 功能说明

- 算子功能：完成 Group Normalization 的反向。
- 计算公式：

$$
\hat{x} = (x - mean) \cdot rstd
$$

$$
s_1 = \sum(dy \cdot gamma), \quad s_2 = \sum(dy \cdot gamma \cdot \hat{x})
$$

$$
dx = \frac{rstd}{M} \cdot gamma \cdot (M \cdot dy - s_1 - \hat{x} \cdot s_2)
$$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnGroupNormalizationGradGetWorkspaceSize"接口获取入参并根据计算流程计算所需workspace大小，再调用"aclnnGroupNormalizationGrad"接口执行计算。

```Cpp
aclnnStatus aclnnGroupNormalizationGradGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *dy,
  const aclTensor *gamma,
  const aclTensor *mean,
  const aclTensor *rstd,
  aclTensor       *dx,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGroupNormalizationGrad(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnGroupNormalizationGradGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1260px"><colgroup>
  <col style="width: 101px">
  <col style="width: 115px">
  <col style="width: 150px">
  <col style="width: 230px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>公式中的 x（前向输入）。</td>
      <td>dtype 需与 dy 保持一致。shape 需与 dy 相同。</td>
      <td>FLOAT、BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>公式中的 dy（上游梯度）。</td>
      <td>数据类型与 x 的数据类型满足互推导关系。</td>
      <td>FLOAT、BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>已广播到 [N, G, M] 的缩放系数。</td>
      <td>dtype 需与 x 保持一致。shape 需与 x 相同。</td>
      <td>FLOAT、BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>每个 group 的均值，形状为 [N, G]。</td>
      <td>dtype 需与 x 保持一致。</td>
      <td>FLOAT、BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输入</td>
      <td>每个 group 的标准差倒数，形状为 [N, G]。</td>
      <td>dtype 需与 x 保持一致。</td>
      <td>FLOAT、BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>公式中的 dx（输入梯度）。</td>
      <td>dtype 需与 x 相同。shape 需与 x 相等。</td>
      <td>FLOAT、BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回 op 执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。
  第一段接口会完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的 x、dy、gamma、mean 或 rstd 是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>x、dy、gamma、mean 或 rstd 的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x、dy、gamma、mean 或 rstd 的 shape 超过 8 维，或 x、dy、gamma 的 shape 低于 3 维。</td>
    </tr>
    <tr>
      <td>x、dy、gamma 与 dx 数据类型不一致。</td>
    </tr>
    <tr>
      <td>x、dy、gamma 的 shape 不一致。</td>
    </tr>
  </tbody></table>

## aclnnGroupNormalizationGrad

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnGroupNormalizationGradGetWorkspaceSize 获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op 执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的 Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

无。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_group_normalization_grad.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> xShape = {2, 4, 128};
  std::vector<int64_t> meanShape = {2, 4};
  void* xDeviceAddr = nullptr;
  void* dyDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* rstdDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* dy = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* rstd = nullptr;
  aclTensor* dx = nullptr;

  auto size = GetShapeSize(xShape);
  std::vector<float> xHostData(size);
  std::vector<float> dyHostData(size);
  std::vector<float> gammaHostData(size);
  std::vector<float> dxHostData(size, 0.0f);
  auto meanSize = GetShapeSize(meanShape);
  std::vector<float> meanHostData(meanSize);
  std::vector<float> rstdHostData(meanSize);
  for (int64_t i = 0; i < size; i++) {
    xHostData[i] = static_cast<float>(i % 128) / 128.0f;
    dyHostData[i] = 0.5f;
    gammaHostData[i] = 1.0f;
  }
  for (int64_t i = 0; i < meanSize; i++) {
    meanHostData[i] = 0.0f;
    rstdHostData[i] = 1.0f;
  }

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dyHostData, xShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gammaHostData, xShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(rstdHostData, meanShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dxHostData, xShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnGroupNormalizationGradGetWorkspaceSize(x, dy, gamma, mean, rstd, dx, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormalizationGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnGroupNormalizationGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormalizationGrad failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  size = GetShapeSize(xShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dxDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  aclDestroyTensor(x);
  aclDestroyTensor(dy);
  aclDestroyTensor(gamma);
  aclDestroyTensor(mean);
  aclDestroyTensor(rstd);
  aclDestroyTensor(dx);
  aclrtFree(xDeviceAddr);
  aclrtFree(dyDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(meanDeviceAddr);
  aclrtFree(rstdDeviceAddr);
  aclrtFree(dxDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
