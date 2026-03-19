# aclnnApplyAdam

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/optim/apply_adam)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- **接口功能：** 实现Adam优化器功能。

- **计算公式：**

  $$
  lr_t = lr \times \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
  $$

  $$
  m_t = \beta_1 \times m_{t-1} + (1 - \beta_1) \times g_t
  $$

  $$
  v_t = \beta_2 \times v_{t-1} + (1 - \beta_2) \times g_t^2
  $$

  若 use_nesterov = true:
  $$
  var_t = var_{t-1} - lr_t \times \frac{\beta_1 \times m_t + (1 - \beta_1) \times g_t}{\sqrt{v_t} + \epsilon}
  $$

  若 use_nesterov = false:
  $$
  var_t = var_{t-1} - lr_t \times \frac{m_t}{\sqrt{v_t} + \epsilon}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnApplyAdamGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnApplyAdam"接口执行计算。

```Cpp
aclnnStatus aclnnApplyAdamGetWorkspaceSize(
    aclTensor *varRef,
    const aclTensor *m,
    const aclTensor *v,
    const aclTensor *beta1Power,
    const aclTensor *beta2Power,
    const aclTensor *lr,
    const aclTensor *beta1,
    const aclTensor *beta2,
    const aclTensor *epsilon,
    const aclTensor *grad,
    bool useLocking,
    bool useNesterov,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnApplyAdam(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnApplyAdamGetWorkspaceSize

- **参数说明：**

  </style>
  <table class="tg" style="undefined;table-layout: fixed; width: 1428px"><colgroup>
  <col style="width: 230px">
  <col style="width: 120px">
  <col style="width: 330px">
  <col style="width: 230px">
  <col style="width: 138px">
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
      <td class="tg-0lax">varRef（aclTensor*）</td>
      <td class="tg-0lax">输入/输出</td>
      <td class="tg-0lax">待计算的权重输入同时也是输出，公式中的var。</td>
      <td class="tg-0lax"></td>
      <td class="tg-0lax">FLOAT16、BFLOAT16、FLOAT32</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">m（aclTensor*）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">Adam优化器中m参数，公式中的m。</td>
      <td class="tg-0lax">shape要求与输入varRef保持一致。</td>
      <td class="tg-0lax">与varRef保持一致</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">v（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">Adam优化器中v参数，公式中的v。</td>
      <td class="tg-0pky">shape要求与输入varRef保持一致。</td>
      <td class="tg-0pky">与varRef保持一致</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta1Power（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">beta1的t次幂。</td>
      <td class="tg-0pky">shape要求为[1]。</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta2Power（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">beta2的t次幂。</td>
      <td class="tg-0pky">shape要求为[1]。</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">lr（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">学习率，公式中的lr。</td>
      <td class="tg-0pky">shape要求为[1]。</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta1（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">beta1参数。</td>
      <td class="tg-0pky">shape要求为[1]。</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta2（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">beta2参数。</td>
      <td class="tg-0pky">shape要求为[1]。</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">epsilon（aclTensor*）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">防止除数为0。</td>
      <td class="tg-0lax">shape要求为[1]。</td>
      <td class="tg-0lax">FLOAT32</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">grad（aclTensor*）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">梯度数据，公式中的g</td>
      <td class="tg-0lax">shape要求与输入varRef保持一致。</td>
      <td class="tg-0lax">与varRef保持一致</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">useLocking（bool）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">是否使用锁机制。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">bool</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">useNesterov（bool）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">是否使用Nesterov momentum。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">bool</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize（uint64_t*）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">返回需要在Device侧申请的workspace大小。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor（aclOpExecutor**）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">返回op执行器，包含了算子计算流程。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  </style>
  <table class="tg" style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 270px">
  <col style="width: 130px">
  <col style="width: 750px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">返回码</th>
      <th class="tg-0pky">错误码</th>
      <th class="tg-0pky">描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">传入的计算输入参数是空指针时。</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="5">161002</td>
      <td class="tg-0pky">传入的计算输入的数据类型不在支持的范围内时。</td>
    </tr>
    <tr>
      <td class="tg-0pky">传入的计算输入的数据类型不一致时。</td>
    </tr>
    <tr>
      <td class="tg-0pky">传入的计算输入的shape不一致时。</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta1Power、beta2Power、lr、beta1、beta2、epsilon的数据类型不为FLOAT32时。</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta1Power、beta2Power、lr、beta1、beta2、epsilon的shape大小不为1时。</td>
    </tr>
  </tbody>
  </table>

## aclnnApplyAdam

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
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnApplyAdamGetWorkspaceSize获取。</td>
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

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算： 
  - aclnnApplyAdam默认确定性实现。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_adam.h"

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

  std::vector<int64_t> varShape = {2, 2};
  std::vector<int64_t> mShape = {2, 2};
  std::vector<int64_t> vShape = {2, 2};
  std::vector<int64_t> beta1PowerShape = {1};
  std::vector<int64_t> beta2PowerShape = {1};
  std::vector<int64_t> lrShape = {1};
  std::vector<int64_t> beta1Shape = {1};
  std::vector<int64_t> beta2Shape = {1};
  std::vector<int64_t> epsilonShape = {1};
  std::vector<int64_t> gradShape = {2, 2};
  void* varDeviceAddr = nullptr;
  void* mDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* beta1PowerDeviceAddr = nullptr;
  void* beta2PowerDeviceAddr = nullptr;
  void* lrDeviceAddr = nullptr;
  void* beta1DeviceAddr = nullptr;
  void* beta2DeviceAddr = nullptr;
  void* epsilonDeviceAddr = nullptr;
  void* gradDeviceAddr = nullptr;
  aclTensor* var = nullptr;
  aclTensor* m = nullptr;
  aclTensor* v = nullptr;
  aclTensor* beta1Power = nullptr;
  aclTensor* beta2Power = nullptr;
  aclTensor* lr = nullptr;
  aclTensor* beta1 = nullptr;
  aclTensor* beta2 = nullptr;
  aclTensor* epsilon = nullptr;
  aclTensor* grad = nullptr;
  std::vector<float> varHostData = {0, 1, 2, 3};
  std::vector<float> mHostData = {0, 1, 2, 3};
  std::vector<float> vHostData = {0, 1, 2, 3};
  std::vector<float> beta1PowerHostData = {0.431};
  std::vector<float> beta2PowerHostData = {0.992};
  std::vector<float> lrHostData = {0.001};
  std::vector<float> beta1HostData = {0.9};
  std::vector<float> beta2HostData = {0.999};
  std::vector<float> epsilonHostData = {1e-8};
  std::vector<float> gradHostData = {0, 1, 2, 3};
  bool useLocking = false;
  bool useNesterov = false;

  ret = CreateAclTensor(varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(mHostData, mShape, &mDeviceAddr, aclDataType::ACL_FLOAT, &m);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(beta1PowerHostData, beta1PowerShape, &beta1PowerDeviceAddr, aclDataType::ACL_FLOAT, &beta1Power);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(beta2PowerHostData, beta2PowerShape, &beta2PowerDeviceAddr, aclDataType::ACL_FLOAT, &beta2Power);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lrHostData, lrShape, &lrDeviceAddr, aclDataType::ACL_FLOAT, &lr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(beta1HostData, beta1Shape, &beta1DeviceAddr, aclDataType::ACL_FLOAT, &beta1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(beta2HostData, beta2Shape, &beta2DeviceAddr, aclDataType::ACL_FLOAT, &beta2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(epsilonHostData, epsilonShape, &epsilonDeviceAddr, aclDataType::ACL_FLOAT, &epsilon);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnApplyAdamGetWorkspaceSize(var, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad, useLocking, useNesterov, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  ret = aclnnApplyAdam(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdam failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  auto size = GetShapeSize(varShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), varDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  aclDestroyTensor(var);
  aclDestroyTensor(m);
  aclDestroyTensor(v);
  aclDestroyTensor(beta1Power);
  aclDestroyTensor(beta2Power);
  aclDestroyTensor(lr);
  aclDestroyTensor(beta1);
  aclDestroyTensor(beta2);
  aclDestroyTensor(epsilon);
  aclDestroyTensor(grad);

  aclrtFree(varDeviceAddr);
  aclrtFree(mDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(beta1PowerDeviceAddr);
  aclrtFree(beta2PowerDeviceAddr);
  aclrtFree(lrDeviceAddr);
  aclrtFree(beta1DeviceAddr);
  aclrtFree(beta2DeviceAddr);
  aclrtFree(epsilonDeviceAddr);
  aclrtFree(gradDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
