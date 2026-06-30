# aclnnApplyFusedEmaAdam

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/optim/apply_fused_ema_adam)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    ×    |
| <term>Atlas training series products</term>                             |    ×    |

## Function

- **Description**: Implements the FusedEmaAdam fusion optimizer.
- **Formula**:

  $$
  (correction_{\beta_1},correction_{\beta_2},)=\begin{cases}
  (1,1),&biasCorrection=False\\
  (1-\beta_1^{step},1-\beta_2^{step}),&biasCorrection=True
  \end{cases}
  $$
  
  $$
  grad=\begin{cases}
  grad+weightDecay*var,&mode=0\\
  grad,&mode=1
  \end{cases}
  $$
  
  $$
  m_{out}=\beta_1*m+(1-\beta_1)*grad
  $$

  $$
  v_{out}=\beta_2*v+(1-\beta_2)*grad^2
  $$

  $$
  m_{next}=m_{out}/correction_{\beta_1}
  $$

  $$
  v_{next}=v_{out}/correction_{\beta_2}
  $$

  $$
  denom=\sqrt{v_{next}}+eps
  $$

  $$
  update=\begin{cases}
  m_{next}/denom,&mode=0\\
  m_{next}/denom+weightDecay*var,&mode=1
  \end{cases}
  $$

  $$
  var_{out}=var-lr*update
  $$

  $$
  s_{out}=emaDecay*s+(1-emaDecay)*var_{out}
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnApplyFusedEmaAdamGetWorkspaceSize** is called to obtain the input, the workspace size required for computation, and the executor that contains the operator computation process. Then, **aclnnApplyFusedEmaAdam** is called to perform computation.

* `aclnnStatus aclnnApplyFusedEmaAdamGetWorkspaceSize(const aclTensor* grad, aclTensor* varRef, aclTensor* mRef, aclTensor* vRef, aclTensor* sRef, const aclTensor* step, double lr, double emaDecay, double beta1, double beta2, double eps, int64_t mode, bool biasCorrection, double weightDecay, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnApplyFusedEmaAdam(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnApplyFusedEmaAdamGetWorkspaceSize

- **Parameters:**
  - **grad** (aclTensor*, compute input): gradient of the parameter to be updated, corresponding to `grad` in the formula. It is an aclTensor on the device. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **varRef** (aclTensor\*, compute input/output): parameter to be updated, corresponding to `var` in the formula. It is an aclTensor on the device. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The shape and data type must be the same as those of **grad**. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **mRef** (aclTensor\*, compute input/output): first-order momentum of the parameter to be updated, corresponding to `m` in the formula. It is an aclTensor on the device. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The shape and data type must be the same as those of **grad**. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **vRef** (aclTensor\*, compute input/output): second-order momentum corresponding to the parameter to be updated, corresponding to `v` in the formula. It is an aclTensor on the device. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The shape and data type must be the same as those of **grad**. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **sRef** (aclTensor\*, compute input/output): EMA weight corresponding to the parameter to be updated, corresponding to `s` in the formula. It is an aclTensor on the device. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The shape and data type must be the same as those of **grad**. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **step** (aclTensor*, compute input): number of updates of the optimizer, corresponding to `step` in the formula. It is an aclTensor on the device. The data type can be INT64. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **lr** (double, compute input): learning rate, corresponding to `lr` in the formula.
  - **emaDecay** (double, compute input): decay rate of exponential moving average (EMA), corresponding to `emaDecay` in the formula.
  - **beta1** (double, compute input): coefficient for calculating the first-order momentum, corresponding to $\beta_1$ in the formula.
  - **beta2** (double, compute input): coefficient for calculating the second-order momentum, corresponding to $\beta_2$ in the formula.
  - **eps** (double, compute input): added to the denominator for numerical stability, corresponding to `eps` in the formula.
  - **mode** (int64_t, compute input): controls whether to apply L2 regularization or weight decay, corresponding to `mode` in the formula. The value **1** indicates adamw, and the value **0** indicates L2.
  - **biasCorrection** (bool, compute input): controls whether to correct the bias, corresponding to `biasCorrection` in the formula. The value **true** indicates that the correction is performed, and the value **false** indicates that the correction is not performed.
  - **weightDecay** (double, compute input): weight decay, corresponding to `weightDecay` in the formula.
  - **workspaceSize** (uint64\_t\*, output): size of the workspace to be allocated on the device.
  - **executor** (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  ```
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input or output tensor is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of the input or output is not supported.
                                       2. The data types and shapes of the input grad, var, m, v, and s are different.
  ```

## aclnnApplyFusedEmaAdam

- **Parameters:**
  - **workspace** (void\*, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API **aclnnApplyFusedEmaAdamGetWorkspaceSize**.
  - **executor** (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**

  aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

The data types and shapes of the input grad, var, m, v, and s must be the same.

- Deterministic compute:
  - **aclnnApplyFusedEmaAdam** defaults to a deterministic implementation.


## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_fused_ema_adam.h"
#include <iostream>
#include <vector>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // (Fixed writing) Initialize resources.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // Call aclrtMemcpy to copy the data from the host to the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct the input and output based on the API.
  // input
  std::vector<float> gradHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> varHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> mHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> vHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> sHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> stepHostData = {10, 10, 10, 10};
  std::vector<int64_t> inputShape = {2, 2, 2};
  std::vector<int64_t> stepShape = {2, 2};
  void *gradDeviceAddr = nullptr;
  void *varDeviceAddr = nullptr;
  void *mDeviceAddr = nullptr;
  void *vDeviceAddr = nullptr;
  void *sDeviceAddr = nullptr;
  void *stepDeviceAddr = nullptr;
  aclTensor *grad = nullptr;
  aclTensor *var = nullptr;
  aclTensor *m = nullptr;
  aclTensor *v = nullptr;
  aclTensor *s = nullptr;
  aclTensor *step = nullptr;
  ret = CreateAclTensor(gradHostData, inputShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(varHostData, inputShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(mHostData, inputShape, &mDeviceAddr, aclDataType::ACL_FLOAT, &m);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, inputShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sHostData, inputShape, &sDeviceAddr, aclDataType::ACL_FLOAT, &s);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(stepHostData, stepShape, &stepDeviceAddr, aclDataType::ACL_INT64, &step);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // out, inplace
  std::vector<int64_t> outShape = {2, 2, 2};
  
  // attr
  float lr = 0.001f;
  float emaDecay = 0.5f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  int64_t mode = 1;
  bool bias = true;
  float weightDecay = 0.5f;

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;
  
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  // Call the first-phase API of aclnnApplyFusedEmaAdam.
  ret = aclnnApplyFusedEmaAdamGetWorkspaceSize(grad, var, m, v, s, step, lr, emaDecay, beta1, beta2, eps,
                                               mode, bias, weightDecay, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnApplyFusedEmaAdamGetWorkspaceSize failed. ERROR: %d\n",
                ret);
      return ret);

  // Allocate device memory based on workspaceSize calculated by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the second-phase API of aclnnApplyFusedEmaAdam.
  ret = aclnnApplyFusedEmaAdam(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnApplyFusedEmaAdam failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  PrintOutResult(outShape, &varDeviceAddr);
  PrintOutResult(outShape, &mDeviceAddr);
  PrintOutResult(outShape, &vDeviceAddr);
  PrintOutResult(outShape, &sDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(grad);
  aclDestroyTensor(var);
  aclDestroyTensor(m);
  aclDestroyTensor(v);
  aclDestroyTensor(s);
  aclDestroyTensor(step);

  // 7. Release device resources.
  aclrtFree(gradDeviceAddr);
  aclrtFree(varDeviceAddr);
  aclrtFree(mDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(sDeviceAddr);
  aclrtFree(stepDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
