# aclnnApplyAdamW

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/optim/apply_adam_w)

## Supported Products
| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    ×    |
| <term>Atlas training series products</term>                             |    ×    |



## Function

- **Description**: Implements the AdamW optimizer.

- **Formula**:

  $$
  g_t=\begin{cases}-g_t
  & \text{ if } maxmize= true\\
  g_t  & \text{ if } maxmize=false
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

* `aclnnStatus aclnnApplyAdamWGetWorkspaceSize(aclTensor* varRef, aclTensor* mRef, aclTensor* vRef, const aclTensor* beta1Power, const aclTensor* beta2Power, const aclTensor* lr, const aclTensor* weightDecay, const aclTensor* beta1, const aclTensor* beta2, const aclTensor* eps, const aclTensor* grad, const aclTensor* maxGradNormOptional, bool amsgrad, bool maximize, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnApplyAdamW(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnApplyAdamWGetWorkspaceSize

- **Parameters:**

  * **varRef** (aclTensor\*, compute input/compute output): weight input and output (theta in the formula), which is an aclTensor on the device. The shape support 1D to 8D. The data type can be FLOAT16, BFLOAT16, or FLOAT32. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **mRef** (aclTensor\*, compute input/compute output): m parameter in the AdamW optimizer (m in the formula), which is an aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The shape and dtype must be the same as those of **varRef**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **vRef** (aclTensor\*, compute input/compute output): v parameter in the AdamW optimizer (v in the formula), which is an aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The shape and dtype must be the same as those of **varRef**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **beta1Power** (aclTensor\*, compute input): beta1^(t-1) parameter, which is an aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The shape must be [1], and the dtype must be the same as that of **varRef**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **beta2Power** (aclTensor\*, compute input): beta2^(t-1) parameter, which is an aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The shape must be [1], and the dtype must be the same as that of **varRef**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **lr** (aclTensor\*, compute input): learning rate (eta in the formula), which is an aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The shape must be [1], and the dtype must be the same as that of **varRef**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **weightDecay** (aclTensor\*, compute input): weight decay coefficient, which is an aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The shape must be [1], and the dtype must be the same as that of **varRef**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **beta1** (aclTensor\*, compute input): beta1 parameter, which is an aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The shape must be [1], and the dtype must be the same as that of **varRef**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **beta2** (aclTensor\*, compute input): beta2 parameter, which is an aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The shape must be [1], and the dtype must be the same as that of **varRef**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **eps** (aclTensor\*, compute input): parameter for avoiding division by zero, which is an aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The shape must be [1], and the dtype must be the same as that of **varRef**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **grad** (aclTensor*, compute input): gradient data (g_t in the formula), which is an aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The shape and dtype must be the same as those of **varRef**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **maxGradNormOptional** (aclTensor\*, compute input): stores the maximum value of the v parameter (v in the formula), which is an aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32. The shape and dtype must be the same as those of **varRef**. This parameter is mandatory when **amsgrad** is set to **true** and optional when **amsgrad** is set to **false**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **amsgrad** (bool, compute input): whether to use the maxGradNormOptional variable. The data type is BOOL.
  * **maximize** (bool, compute input): whether to reverse the gradient. The gradient ascent direction is used to optimize the weight to maximize the loss function. The data type is BOOL.
  * **workspaceSize** (uint64_t\*, output): size of the workspace to be allocated on the device.
  * **executor** (aclOpExecutor\*\*, output): memory address.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input parameter for computation is a null pointer.
                                    2. amsgrad is set to true, and maxGradNormOptional is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The input data type for computation is not supported.
                                    2. The input data types for computation are inconsistent.
                                    3. The input shapes for computation are inconsistent.
                                    4. amsgrad is set to true, and the data type or shape of maxGradNormOptional is different from that of varRef.
                                    5. The shape size of beta1Power, beta2Power, lr, weightDecay, beta1, beta2, or eps is not 1.
  ```

## aclnnApplyAdamW

- **Parameters:**

  * **workspace** (void \*, input): address of the workspace to be allocated on the device.
  * **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API **aclnnApplyAdamWGetWorkspaceSize**.
  * **executor** (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  * **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- The data types of the input tensors must be the same. The data type can be FLOAT16, BFLOAT16, or FLOAT32.
- The shape sizes of the input tensors beta1Power, beta2Power, lr, weightDecay, beta1, beta2, and eps should be 1.
- When the input Boolean value of **maximize** is **true**, the **maxGradNormOptional** parameter is mandatory, and the data type and shape must be the same as those of **varRef**.

- Deterministic compute:
  - **aclnnApplyAdamW** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

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
  // (Fixed writing) Initialize resources.
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
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
