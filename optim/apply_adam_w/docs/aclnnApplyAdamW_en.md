# aclnnApplyAdamW

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/optim/apply_adam_w)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     √    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     ×    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- **API function**: Implements the AdamW optimizer.

- **Formula**:

  $$
  g_t=\begin{cases}-g_t
  & \text{ if } maximize= true\\
  g_t  & \text{ if } maximize=false
  \end{cases}
  $$

  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\
  $$

  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$

  $$
  \beta_{1}^{t}=\beta_{1}^{t-1}\times\beta_{1}
  $$

  $$
  \beta_{2}^{t}=\beta_{2}^{t-1}\times\beta_{2}
  $$

  $$
  v_t=\begin{cases}\max(maxGradNorm, v_t)
  & \text{ if } amsgrad = true\\
  v_t  & \text{ if } amsgrad = false
  \end{cases}
  $$

  $$
  \hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}} \\
  $$

  $$
  \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}} \\
  $$

  $$
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnApplyAdamWGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnApplyAdamW** is called to perform computation.

```Cpp
aclnnStatus aclnnApplyAdamWGetWorkspaceSize(
    aclTensor*       varRef, 
    aclTensor*       mRef, 
    aclTensor*       vRef,
    const aclTensor* beta1Power, 
    const aclTensor* beta2Power, 
    const aclTensor* lr,
    const aclTensor* weightDecay, 
    const aclTensor* beta1, 
    const aclTensor* beta2,
    const aclTensor* eps, 
    const aclTensor* grad, 
    const aclTensor* maxGradNormOptional,
    bool             amsgrad, 
    bool             maximize,
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnApplyAdamW(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnApplyAdamWGetWorkspaceSize

- **Parameters:**

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
      <th class="tg-0pky">Parameter Name</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Usage Description</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-continuous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0lax">varRef (aclTensor*) </td>
      <td class="tg-0lax">Input/Output</td>
      <td class="tg-0lax">Weight input to be calculated, which is also the output and θ in the formula.</td>
      <td class="tg-0lax"></td>
      <td class="tg-0lax">FLOAT16, BFLOAT16, FLOAT32</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">mRef (aclTensor*) </td>
      <td class="tg-0lax">Input/Output</td>
      <td class="tg-0lax">Parameter m in the AdamW optimizer, which is m in the formula.</td>
      <td class="tg-0lax">The shape must be the same as that of the input varRef.</td>
      <td class="tg-0lax">Same as varRef</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">vRef (aclTensor*) </td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Parameter v in the AdamW optimizer, which is v in the formula.</td>
      <td class="tg-0pky">The shape must be the same as that of the input varRef.</td>
      <td class="tg-0pky">Same as varRef</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta1Power (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">beta1^(t-1) parameter.</td>
      <td class="tg-0pky">The shape must be [1].</td>
      <td class="tg-0pky">The value must be the same as that of varRef.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta2Power (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">beta2^(t-1) parameter.</td>
      <td class="tg-0pky">The shape must be [1].</td>
      <td class="tg-0pky">The value must be the same as that of varRef.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">lr (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Learning rate, which is η in the formula. Generally, the value is 1e-3, 1e-6, or 1e-9.</td>
      <td class="tg-0pky">The shape must be [1].</td>
      <td class="tg-0pky">The value must be the same as that of varRef.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">weightDecay (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Weight decay coefficient.</td>
      <td class="tg-0pky">The shape must be [1].</td>
      <td class="tg-0pky">The value must be the same as that of varRef.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta1 (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">beta1 parameter.</td>
      <td class="tg-0pky">The shape must be [1].</td>
      <td class="tg-0pky">The value must be the same as that of varRef.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta2 (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">beta2 parameter.</td>
      <td class="tg-0pky">The shape must be [1].</td>
      <td class="tg-0pky">Same as varRef</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">eps (aclTensor*) </td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Prevents the divisor from being 0.</td>
      <td class="tg-0lax">The shape must be [1].</td>
      <td class="tg-0lax">Same as varRef</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">grad (aclTensor*) </td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Gradient data, g</td> in the formula
      <td class="tg-0lax">The shape must be the same as that of the input varRef.</td>
      <td class="tg-0lax">Same as varRef</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">maxGradNormOptional (aclTensor*) </td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">If amsgrad is set to true, the maximum value of maxGradNorm and v is saved in vRef. If amsgrad is set to false, no additional calculation is performed on the value in vRef. This parameter is mandatory when amsgrad is set to true and optional when amsgrad is set to false.</td>
      <td class="tg-0lax">The shape must be the same as that of the input varRef.</td>
      <td class="tg-0lax">The value is the same as that of varRef.</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">1-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">amsgrad (bool) </td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Whether to use the maxGradNormOptional variable.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">bool</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">maximize (bool) </td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Whether to negate the gradient. The weight is optimized in the gradient ascent direction to maximize the loss function.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">bool</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize (uint64_t*) </td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the size of the workspace to be allocated on the device.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor (aclOpExecutor**) </td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the operator executor, including the operator computation process.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table class="tg" style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 270px">
  <col style="width: 130px">
  <col style="width: 750px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Return Code</th>
      <th class="tg-0pky">Error Code</th>
      <th class="tg-0pky">Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky" rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky" rowspan="2">161001</td>
      <td class="tg-0pky">The input parameter for computation is a null pointer.</td>
    </tr>
    <tr>
      <td class="tg-0pky">When amsgrad is true, maxGradNormOptional is a null pointer.</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="5">161002</td>
      <td class="tg-0pky">The input data type of the computation is not supported.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The input data types of the computation are inconsistent.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The input shapes of the computation are inconsistent.</td>
    </tr>
    <tr>
      <td class="tg-0pky">When amsgrad is set to true, the data types or shapes of maxGradNormOptional and varRef are inconsistent.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shapes of beta1Power, beta2Power, lr, weightDecay, beta1, beta2, and eps are not 1.</td>
    </tr>
  </tbody>
  </table>

## aclnnApplyAdamW

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnApplyAdamWGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - **aclnnApplyAdamW** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_adam_w.h"

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
  // (Fixed writing) Initialize AscendCL.
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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy the data from the host to the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> varShape = {2, 2};
  std::vector<int64_t> mShape = {2, 2};
  std::vector<int64_t> vShape = {2, 2};
  std::vector<int64_t> beta1PowerShape = {1};
  std::vector<int64_t> beta2PowerShape = {1};
  std::vector<int64_t> lrShape = {1};
  std::vector<int64_t> weightDecayShape = {1};
  std::vector<int64_t> beta1Shape = {1};
  std::vector<int64_t> beta2Shape = {1};
  std::vector<int64_t> epsShape = {1};
  std::vector<int64_t> gradShape = {2, 2};
  std::vector<int64_t> maxgradShape = {2, 2};
  void* varDeviceAddr = nullptr;
  void* mDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* beta1PowerDeviceAddr = nullptr;
  void* beta2PowerDeviceAddr = nullptr;
  void* lrDeviceAddr = nullptr;
  void* weightDecayDeviceAddr = nullptr;
  void* beta1DeviceAddr = nullptr;
  void* beta2DeviceAddr = nullptr;
  void* epsDeviceAddr = nullptr;
  void* gradDeviceAddr = nullptr;
  void* maxgradDeviceAddr = nullptr;
  aclTensor* var = nullptr;
  aclTensor* m = nullptr;
  aclTensor* v = nullptr;
  aclTensor* beta1Power = nullptr;
  aclTensor* beta2Power = nullptr;
  aclTensor* lr = nullptr;
  aclTensor* weightDecay = nullptr;
  aclTensor* beta1 = nullptr;
  aclTensor* beta2 = nullptr;
  aclTensor* eps = nullptr;
  aclTensor* grad = nullptr;
  aclTensor* maxgrad = nullptr;
  std::vector<float> varHostData = {0, 1, 2, 3};
  std::vector<float> mHostData = {0, 1, 2, 3};
  std::vector<float> vHostData = {0, 1, 2, 3};
  std::vector<float> beta1PowerHostData = {0.431};
  std::vector<float> beta2PowerHostData = {0.992};
  std::vector<float> lrHostData = {0.001};
  std::vector<float> weightDecayHostData = {0.01};
  std::vector<float> beta1HostData = {0.9};
  std::vector<float> beta2HostData = {0.999};
  std::vector<float> epsHostData = {1e-8};
  std::vector<float> gradHostData = {0, 1, 2, 3};
  std::vector<float> maxgradHostData = {0, 1, 2, 3};
  bool amsgrad = true;
  bool maximize = true;
  // Create a var aclTensor.
  ret = CreateAclTensor(varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an m aclTensor.
  ret = CreateAclTensor(mHostData, mShape, &mDeviceAddr, aclDataType::ACL_FLOAT, &m);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  //Create a v aclTensor.
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a beta1Power aclTensor.
  ret = CreateAclTensor(beta1PowerHostData, beta1PowerShape, &beta1PowerDeviceAddr, aclDataType::ACL_FLOAT, &beta1Power);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a beta2Power aclTensor.
  ret = CreateAclTensor(beta2PowerHostData, beta2PowerShape, &beta2PowerDeviceAddr, aclDataType::ACL_FLOAT, &beta2Power);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an lr aclTensor.
  ret = CreateAclTensor(lrHostData, lrShape, &lrDeviceAddr, aclDataType::ACL_FLOAT, &lr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a weightDecay aclTensor.
  ret = CreateAclTensor(weightDecayHostData, weightDecayShape, &weightDecayDeviceAddr, aclDataType::ACL_FLOAT, &weightDecay);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a beta1 aclTensor.
  ret = CreateAclTensor(beta1HostData, beta1Shape, &beta1DeviceAddr, aclDataType::ACL_FLOAT, &beta1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a beta2 aclTensor.
  ret = CreateAclTensor(beta2HostData, beta2Shape, &beta2DeviceAddr, aclDataType::ACL_FLOAT, &beta2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an eps aclTensor.
  ret = CreateAclTensor(epsHostData, epsShape, &epsDeviceAddr, aclDataType::ACL_FLOAT, &eps);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a grad aclTensor.
  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a maxgrad aclTensor.
  ret = CreateAclTensor(maxgradHostData, maxgradShape, &maxgradDeviceAddr, aclDataType::ACL_FLOAT, &maxgrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnApplyAdamW.
  ret = aclnnApplyAdamWGetWorkspaceSize(var, m, v, beta1Power, beta2Power, lr, weightDecay, beta1, beta2, eps, grad, maxgrad, amsgrad, maximize, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnApplyAdamW.
  ret = aclnnApplyAdamW(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamW failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(varShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), varDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(var);
  aclDestroyTensor(m);
  aclDestroyTensor(v);
  aclDestroyTensor(beta1Power);
  aclDestroyTensor(beta2Power);
  aclDestroyTensor(lr);
  aclDestroyTensor(weightDecay);
  aclDestroyTensor(beta1);
  aclDestroyTensor(beta2);
  aclDestroyTensor(eps);
  aclDestroyTensor(grad);
  aclDestroyTensor(maxgrad);

  // 7. Release device resources.
  aclrtFree(varDeviceAddr);
  aclrtFree(mDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(beta1PowerDeviceAddr);
  aclrtFree(beta2PowerDeviceAddr);
  aclrtFree(lrDeviceAddr);
  aclrtFree(weightDecayDeviceAddr);
  aclrtFree(beta1DeviceAddr);
  aclrtFree(beta2DeviceAddr);
  aclrtFree(epsDeviceAddr);
  aclrtFree(gradDeviceAddr);
  aclrtFree(maxgradDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
