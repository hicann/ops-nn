# aclnnReluGradV3


## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

计算ReLU函数的梯度。前向ReLU定义为 $z = \max(0, x)$。本算子实现该激活函数对输入 $x$ 的反向梯度计算：$grad\_input = (x > 0) \ ? \ grad\_output : 0$。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnReluGradV3GetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnReluGradV3"接口执行计算。

```Cpp
aclnnStatus aclnnReluGradV3GetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* y,
    aclTensor*       z,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnReluGradV3(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnReluGradV3GetWorkspaceSize

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
        <td class="tg-0pky">x（aclTensor*）</td>
        <td class="tg-0pky">输入</td>
        <td class="tg-0pky">前向ReLU的输入。</td>
        <td class="tg-0pky">shape决定输出shape。<br>数据类型需为FLOAT / FLOAT16 / BFLOAT16 / INT32 / UINT8 / INT8 / BFLOAT16。</td>
        <td class="tg-0pky">FLOAT / FLOAT16 / BFLOAT16 / INT32 / UINT8 / INT8</td>
        <td class="tg-0pky">ND</td>
        <td class="tg-0pky">1-8</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">y（aclTensor*）</td>
        <td class="tg-0pky">输入</td>
        <td class="tg-0pky">反向传播的梯度输入（grad_output）。</td>
        <td class="tg-0pky">支持与x shape一致，或作为单元素标量广播到x的shape。<br>数据类型需为FLOAT / FLOAT16 / BFLOAT16 / INT32 / UINT8 / INT8 / BFLOAT16。</td>
        <td class="tg-0pky">FLOAT / FLOAT16 / BFLOAT16 / INT32 / UINT8 / INT8</td>
        <td class="tg-0pky">ND</td>
        <td class="tg-0pky">1-8</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">z（aclTensor*）</td>
        <td class="tg-0pky">输出</td>
        <td class="tg-0pky">梯度计算结果输出（grad_input）。</td>
        <td class="tg-0pky">shape与x一致。</td>
        <td class="tg-0pky">FLOAT / FLOAT16 / BFLOAT16 / INT32 / UINT8 / INT8</td>
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
        <td class="tg-0pky">传入的x、y或z是空指针。</td>
      </tr>
      <tr>
        <td class="tg-0pky" rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td class="tg-0pky" rowspan="2">161002</td>
        <td class="tg-0pky">x、y或z的数据类型不在支持的范围之内。</td>
      </tr>
      <tr>
        <td class="tg-0lax">x、y和z的shape不一致。</td>
      </tr>
    </tbody>
  </table>

## aclnnReluGradV3

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
          <td>在Device侧申请的workspace大小，由第一段接口aclnnReluGradV3GetWorkspaceSize获取。</td>
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
  - aclnnReluGradV3默认确定性实现。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_relu_grad_v3.h"

#define CHECK_RET(cond, return_expr)
  do {
    if (!(cond)) {
      return_expr;
    }
  } while (0)

#define LOG_PRINT(message, ...)
  do {
    printf(message, ##__VA_ARGS__);
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

  std::vector<int64_t> xShape = {4, 4};
  std::vector<float> xHostData = {0.5f, -0.3f, 2.0f, -1.5f, 0.8f, -0.7f, 1.2f, 0.1f,
                                   1.5f, -2.0f, 0.3f, -0.9f, 0.0f, 1.0f, -1.0f, 0.6f};
  std::vector<int64_t> yShape = {4, 4};
  std::vector<float> yHostData(16, 0.25f);
  std::vector<int64_t> zShape = {4, 4};
  std::vector<float> zHostData(16, 0.0f);

  void* xDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  void* zDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* y = nullptr;
  aclTensor* z = nullptr;

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(zHostData, zShape, &zDeviceAddr, aclDataType::ACL_FLOAT, &z);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  ret = aclnnReluGradV3GetWorkspaceSize(x, y, z, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReluGradV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnReluGradV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReluGradV3 failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 打印结果
  auto size = GetShapeSize(zShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), zDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    float expected = (xHostData[i] > 0.0f) ? yHostData[i] : 0.0f;
    LOG_PRINT("result[%ld]=%f, expected=%f\n", i, resultData[i], expected);
  }

  aclDestroyTensor(x);
  aclDestroyTensor(y);
  aclDestroyTensor(z);

  aclrtFree(xDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(zDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
