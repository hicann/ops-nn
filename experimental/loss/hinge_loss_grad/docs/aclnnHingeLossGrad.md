# aclnnHingeLossGrad


## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend910B4</term>     |     ?    |

## 功能说明

计算hinge loss函数的梯度。Hinge Loss常用于SVM等分类任务中，其前向损失为 $L = \max(0, 1 - y \cdot f(x))$。本算子实现该损失对预测值 $f(x)$ 的反向梯度计算。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnHingeLossGradGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnHingeLossGrad"接口执行计算。

```Cpp
aclnnStatus aclnnHingeLossGradGetWorkspaceSize(
    const aclTensor* predict,
    const aclTensor* target,
    const aclTensor* gradOutput,
    aclTensor*       gradInput,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnHingeLossGrad(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnHingeLossGradGetWorkspaceSize

- **参数说明**：

    </style>
    <table class="tg" style="undefined;table-layout: fixed; width: 1547px"><colgroup>
    <col style="width: 217px">
    <col style="width: 120px">
    <col style="width: 280px">
    <col style="width: 350px">
    <col style="width: 200px">
    <col style="width: 115px">
    <col style="width: 120px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
        <th class="tg-0pky">参数名</th>
        <th class="tg-0pky">输入/输出</th>
        <th class="tg-0pky">描述</th>
        <th class="tg-0pky">使用说明</th>
        <th class="tg-0pky">数据类型</th>
        <th class="tg-0pky">数据格式</th>
        <th class="tg-0pky">维度(shape)</th>
        <th class="tg-0pky">非连续Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td class="tg-0pky">predict（aclTensor*）</td>
        <td class="tg-0pky">输入</td>
        <td class="tg-0pky">模型预测值输入。</td>
        <td class="tg-0pky">shape需与target、gradOutput完全一致。<br>数据类型需为FLOAT。</td>
        <td class="tg-0pky">FLOAT / FLOAT16 / BF16</td>
        <td class="tg-0pky">ND</td>
        <td class="tg-0pky">1-8</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">target（aclTensor*）</td>
        <td class="tg-0pky">输入</td>
        <td class="tg-0pky">真实标签输入，通常取值为+1或-1。</td>
        <td class="tg-0pky">shape需与predict、gradOutput完全一致。<br>数据类型需为FLOAT / FLOAT16 / BF16。</td>
        <td class="tg-0pky">FLOAT / FLOAT16 / BF16</td>
        <td class="tg-0pky">ND</td>
        <td class="tg-0pky">1-8</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">gradOutput（aclTensor*）</td>
        <td class="tg-0pky">输入</td>
        <td class="tg-0pky">反向传播的梯度输入。</td>
        <td class="tg-0pky">shape需与predict、target完全一致。<br>数据类型需为FLOAT / FLOAT16 / BF16。</td>
        <td class="tg-0pky">FLOAT / FLOAT16 / BF16</td>
        <td class="tg-0pky">ND</td>
        <td class="tg-0pky">1-8</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">gradInput（aclTensor*）</td>
        <td class="tg-0pky">输出</td>
        <td class="tg-0pky">梯度计算结果输出。</td>
        <td class="tg-0pky">shape与predict一致。</td>
        <td class="tg-0pky">FLOAT / FLOAT16 / BF16</td>
        <td class="tg-0pky">ND</td>
        <td class="tg-0pky">1-8</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">workspaceSize（uint64_t*）</td>
        <td class="tg-0pky">输出</td>
        <td class="tg-0pky">返回需要在Device侧申请的workspace大小。</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
      </tr>
      <tr>
        <td class="tg-0pky">executor（aclOpExecutor**）</td>
        <td class="tg-0pky">输出</td>
        <td class="tg-0pky">返回op执行器，包含了算子计算流程。</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
      </tr>
    </tbody></table>

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
    </style>
  <table class="tg" style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 269px">
  <col style="width: 135px">
  <col style="width: 746px">
    </colgroup>
    <thead>
      <tr>
        <th class="tg-0pky">返回值</th>
        <th class="tg-0pky">错误码</th>
        <th class="tg-0pky">描述</th>
      </tr></thead>
    <tbody>
      <tr>
        <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
        <td class="tg-0pky">161001</td>
        <td class="tg-0pky">传入的predict、target、gradOutput或gradInput是空指针。</td>
      </tr>
      <tr>
        <td class="tg-0pky" rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td class="tg-0pky" rowspan="2">161002</td>
        <td class="tg-0pky">predict、target、gradOutput或gradInput的数据类型不在支持的范围之内。</td>
      </tr>
      <tr>
        <td class="tg-0lax">predict、target、gradOutput和gradInput的shape不一致。</td>
      </tr>
    </tbody>
  </table>

## aclnnHingeLossGrad

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
          <td>在Device侧申请的workspace内存地址。</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>输入</td>
          <td>在Device侧申请的workspace大小，由第一段接口aclnnHingeLossGradGetWorkspaceSize获取。</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>输入</td>
          <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
          <td>stream</td>
          <td>输入</td>
          <td>指定执行任务的Stream。</td>
        </tr>
      </tbody>
    </table>


- **返回值**：

  **aclnnStatus**：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnHingeLossGrad默认确定性实现。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_hinge_loss_grad.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> predictShape = {32, 4, 4, 4};
  std::vector<float> predictHostData(2048, 0.5f);
  std::vector<int64_t> targetShape = {32, 4, 4, 4};
  std::vector<float> targetHostData(2048, 1.0f);
  std::vector<int64_t> gradOutputShape = {32, 4, 4, 4};
  std::vector<float> gradOutputHostData(2048, 0.25f);
  std::vector<int64_t> gradInputShape = {32, 4, 4, 4};
  std::vector<float> gradInputHostData(2048, 0.0f);

  void* predictDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* gradOutputDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* predict = nullptr;
  aclTensor* targetT = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* gradInput = nullptr;

  ret = CreateAclTensor(predictHostData, predictShape, &predictDeviceAddr, aclDataType::ACL_FLOAT, &predict);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &targetT);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  ret = aclnnHingeLossGradGetWorkspaceSize(predict, targetT, gradOutput, gradInput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHingeLossGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnHingeLossGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHingeLossGrad failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 打印结果
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    float margin = 1.0f - targetHostData[i] * predictHostData[i];
    float expected = (margin > 0.0f) ? (-targetHostData[i] * gradOutputHostData[i]) : 0.0f;
    LOG_PRINT("result[%ld]=%f, expected=%f\n", i, resultData[i], expected);
  }

  aclDestroyTensor(predict);
  aclDestroyTensor(targetT);
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(gradInput);

  aclrtFree(predictDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(gradInputDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
