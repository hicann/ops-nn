# aclnnHuberLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品</term>     |     √    |

## 功能说明

计算预测值 `predictions` 与目标值 `targets` 之间的逐元素 Huber 损失。该算子不执行广播和规约，输出 `loss` 的 shape、数据类型与输入保持一致。

设 $e = predictions - targets$，则逐元素损失为：

$$
loss =
\begin{cases}
0.5e^2, & |e| \leq delta \\
delta(|e| - 0.5delta), & |e| > delta
\end{cases}
$$

对于 `FLOAT16` 和 `BFLOAT16`，Kernel 内部转换为 `FLOAT` 进行计算，再转换回输出数据类型。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnHuberLossGetWorkspaceSize”接口获取计算所需 workspace 大小以及包含算子计算流程的执行器，再调用“aclnnHuberLoss”接口执行计算。

```Cpp
aclnnStatus aclnnHuberLossGetWorkspaceSize(
    const aclTensor* predictions,
    const aclTensor* targets,
    float            delta,
    aclTensor*       loss,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnHuberLoss(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnHuberLossGetWorkspaceSize

- **参数说明**：

  <table class="tg" style="undefined;table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 220px">
  <col style="width: 110px">
  <col style="width: 250px">
  <col style="width: 410px">
  <col style="width: 180px">
  <col style="width: 100px">
  <col style="width: 180px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>predictions（aclTensor*）</td>
      <td>输入</td>
      <td>预测值输入。</td>
      <td>shape 需要与 `targets`、`loss` 完全一致。数据类型需与 `targets`、`loss` 保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>任意维</td>
    </tr>
    <tr>
      <td>targets（aclTensor*）</td>
      <td>输入</td>
      <td>目标值输入。</td>
      <td>shape 需要与 `predictions`、`loss` 完全一致。数据类型需与 `predictions`、`loss` 保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>与 `predictions` 一致</td>
    </tr>
    <tr>
      <td>delta（float）</td>
      <td>输入</td>
      <td>Huber 损失的分段阈值。</td>
      <td>必须大于 `0`。算子属性默认值为 `1.0`；ACLNN 接口调用时需显式传入。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>loss（aclTensor*）</td>
      <td>输出</td>
      <td>逐元素 Huber 损失。</td>
      <td>shape 和数据类型需与 `predictions`、`targets` 保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>与 `predictions` 一致</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回 op 执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**：

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，常见报错场景包括：

  - `predictions`、`targets` 或 `loss` 是空指针。
  - `predictions`、`targets` 或 `loss` 的数据类型不在支持范围内，或三者数据类型不一致。
  - `predictions`、`targets` 与 `loss` 的 shape 不一致。
  - `delta` 小于或等于 `0`。

## aclnnHuberLoss

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 200px">
    <col style="width: 150px">
    <col style="width: 800px">
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
        <td>在 Device 侧申请的 workspace 内存地址。本算子无用户 workspace；仍应以第一段接口返回值为准。</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnHuberLossGetWorkspaceSize 获取。</td>
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

- **返回值**：

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 输入张量 `predictions`、`targets` 与输出张量 `loss` 的 shape 必须完全一致，不支持广播。
- 输入张量 `predictions`、`targets` 与输出张量 `loss` 的数据类型必须一致。
- 数据类型仅支持 `FLOAT`、`FLOAT16`、`BFLOAT16`，数据格式仅支持 `ND`。
- `delta` 必须大于 `0`，算子属性默认值为 `1.0`。
- 不支持 reduction，不需要用户 workspace，无跨核归约。
- 支持动态 rank 和动态 shape。
- 仅支持 Atlas A2 训练系列产品。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。仓库中可运行的完整样例参见 [test_aclnn_huber_loss.cpp](../examples/test_aclnn_huber_loss.cpp)。

```Cpp
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_huber_loss.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t size = 1;
  for (int64_t dim : shape) {
    size *= dim;
  }
  return size;
}

int CreateAclTensor(
  const std::vector<float>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclTensor** tensor)
{
  const size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * sizeof(float);
  auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtMemcpy(*deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(
    shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtCreateStream(&stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  const std::vector<int64_t> shape = {7};
  const std::vector<float> predictionsHost = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0};
  const std::vector<float> targetsHost(predictionsHost.size(), 0.0);
  std::vector<float> lossHost(predictionsHost.size(), 0.0);
  constexpr float delta = 1.0f;

  void* predictionsDevice = nullptr;
  void* targetsDevice = nullptr;
  void* lossDevice = nullptr;
  aclTensor* predictions = nullptr;
  aclTensor* targets = nullptr;
  aclTensor* loss = nullptr;
  ret = CreateAclTensor(predictionsHost, shape, &predictionsDevice, &predictions);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(targetsHost, shape, &targetsDevice, &targets);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lossHost, shape, &lossDevice, &loss);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnHuberLossGetWorkspaceSize(predictions, targets, delta, loss, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  void* workspace = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  ret = aclnnHuberLoss(workspace, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtMemcpy(
    lossHost.data(), lossHost.size() * sizeof(float), lossDevice, lossHost.size() * sizeof(float),
    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::printf("NPU loss: [");
  for (size_t i = 0; i < lossHost.size(); ++i) {
    std::printf(i + 1 == lossHost.size() ? "%.6f" : "%.6f, ", lossHost[i]);
  }
  std::printf("]\n");

  aclDestroyTensor(predictions);
  aclDestroyTensor(targets);
  aclDestroyTensor(loss);
  aclrtFree(predictionsDevice);
  aclrtFree(targetsDevice);
  aclrtFree(lossDevice);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
