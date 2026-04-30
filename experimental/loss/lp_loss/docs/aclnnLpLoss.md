# aclnnLpLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品</term>     |     √    |

## 功能说明

计算预测值与标签值之间的 Lp 范数损失。`reduction` 指定损失的规约方式，支持 `"none"`、`"mean"`、`"sum"`。`"none"` 表示输出逐元素损失，`"mean"` 表示输出所有元素损失的平均值，`"sum"` 表示输出所有元素损失的求和值。

当前仅支持 `p=1`，此时逐元素损失计算公式为：

$$
loss_i = |predict_i - label_i|
$$

对于 `float16` 和 `bfloat16` 类型，内部会转换为 `float32` 进行高精度计算。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnLpLossGetWorkspaceSize”接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用“aclnnLpLoss”接口执行计算。

```Cpp
aclnnStatus aclnnLpLossGetWorkspaceSize(
    const aclTensor* predict,
    const aclTensor* label,
    int64_t          p,
    const char*      reduction,
    const aclTensor* out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnLpLoss(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnLpLossGetWorkspaceSize

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
      <td>predict（aclTensor*）</td>
      <td>输入</td>
      <td>预测值输入。</td>
      <td>shape 需要与 `label` 完全一致。数据类型需与 `label`、`out` 保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>任意维</td>
    </tr>
    <tr>
      <td>label（aclTensor*）</td>
      <td>输入</td>
      <td>标签值输入。</td>
      <td>shape 需要与 `predict` 完全一致。数据类型需与 `predict`、`out` 保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>与 `predict` 一致</td>
    </tr>
    <tr>
      <td>p（int64_t）</td>
      <td>输入</td>
      <td>Lp 范数的阶数。</td>
      <td>当前仅支持 `1`。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction（const char*）</td>
      <td>输入</td>
      <td>指定要应用到输出的规约方式。</td>
      <td>支持 `"none"`、`"mean"`、`"sum"`。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>损失计算结果。</td>
      <td>当 `reduction="none"` 时，`out` 的 shape 需与 `predict` 一致；当 `reduction="mean"` 或 `"sum"` 时，`out` 需为标量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>`predict` 同 shape 或标量</td>
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

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，常见报错场景包括：

  - `predict`、`label` 或 `out` 是空指针。
  - `predict`、`label` 或 `out` 的数据类型不在支持范围内，或三者数据类型不一致。
  - `predict` 与 `label` 的 shape 不一致。
  - `reduction="none"` 时 `out` 的 shape 与输入不一致，或 `reduction="mean"`、`"sum"` 时 `out` 不是标量。
  - `p` 不等于 `1`，或 `reduction` 取值不在支持范围内。

## aclnnLpLoss

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
        <td>在 Device 侧申请的 workspace 内存地址。</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnLpLossGetWorkspaceSize 获取。</td>
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

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 输入张量 `predict` 和 `label` 的 shape 必须完全一致，不支持广播。
- 输入张量 `predict`、`label` 与输出张量 `out` 的数据类型必须一致。
- 数据格式仅支持 `ND`。
- 属性 `p` 当前仅支持 `1`。
- `reduction` 仅支持 `"none"`、`"mean"`、`"sum"`。
- 当 `reduction="none"` 时，输出 `out` 的 shape 与输入一致；当 `reduction="mean"` 或 `"sum"` 时，输出 `out` 为标量。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lp_loss.h"

 #define CHECK_RET(cond, return_expr) \
  do {                              \
    if (!(cond)) {                \
      return_expr;              \
    }                             \
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
  // 固定写法，资源初始化
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

  // 调用 aclrtMalloc 申请 device 侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用 aclrtMemcpy 将 host 侧数据拷贝到 device 侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续 tensor 的 strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用 aclCreateTensor 接口创建 aclTensor
  *tensor = aclCreateTensor(
    shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
    *deviceAddr);
  return 0;
}

int main()
{
  // 1. （固定写法）device/stream 初始化
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出
  std::vector<int64_t> commonShape = {2, 2};
  std::vector<int64_t> yShape = {};

  void* predictDeviceAddr = nullptr;
  void* labelDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;

  aclTensor* predict = nullptr;
  aclTensor* label = nullptr;
  aclTensor* y = nullptr;

  // 构造测试数据
  std::vector<float> predictHostData = {0.0, 1.0, 2.0, 3.0};
  std::vector<float> labelHostData = {2.0, 0.5, 2.0, 2.5};
  std::vector<float> yHostData = {0.0};

  // 创建输入 Tensors
  ret = CreateAclTensor(predictHostData, commonShape, &predictDeviceAddr, aclDataType::ACL_FLOAT, &predict);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(labelHostData, commonShape, &labelDeviceAddr, aclDataType::ACL_FLOAT, &label);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建输出 Tensor，mean 模式输出标量
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 准备 Attribute
  int64_t p = 1;
  const char* reduction = "mean";

  // 3. 调用 CANN 算子库 API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 获取 Workspace 大小
  ret = aclnnLpLossGetWorkspaceSize(predict, label, p, reduction, y, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLpLossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 申请 workspace
  void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

  // 执行算子
  ret = aclnnLpLoss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLpLoss failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出结果
  auto size = GetShapeSize(yShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(
    resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("\nResult Y: \n");
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("Index[%ld]: %f\n", i, resultData[i]);
  }

  // 6. 释放资源
  aclDestroyTensor(predict);
  aclDestroyTensor(label);
  aclDestroyTensor(y);

  aclrtFree(predictDeviceAddr);
  aclrtFree(labelDeviceAddr);
  aclrtFree(yDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
    }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```