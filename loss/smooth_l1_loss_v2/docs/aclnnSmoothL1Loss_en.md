# aclnnSmoothL1Loss

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/smooth_l1_loss_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √       |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |   ×     |
| <term>Atlas training series products</term>                             |   √     |
## Function

- Description: Computes the SmoothL1 loss function.
- Formula:
  
  If `reduction` is `none`, the loss function for batch N is defined as follows:

  $$
  \ell(self,target) = L = \{l_1,\dots,l_N\}^\top
  $$
  
  $l_n$ is calculated as follows:
  
  $$
  l_n = \begin{cases}
  0.5(self_n-target_n)^2/beta, & if |self_n-target_n| < beta \\
  |self_n-target_n| - 0.5*beta, &  otherwise
  \end{cases}
  $$

  If `reduction` is `mean` or `sum`:

  $$
  \ell(self,target)=\begin{cases}
  mean(L), & \text{if reduction} = \text{mean}\\
  sum(L), & \text{if reduction} = \text{sum}
  \end{cases}
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnSmoothL1LossGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnSmoothL1Loss** is called to perform computation.

- `aclnnStatus aclnnSmoothL1LossGetWorkspaceSize(const aclTensor* self, const aclTensor* target, int64_t reduction, float beta, aclTensor* result, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnSmoothL1Loss(void* workspace, uint64_t workspaceSize,  aclOpExecutor* executor, aclrtStream stream)`

## aclnnSmoothL1LossGetWorkspaceSize
- **Parameters:**

  - **self** (aclTensor*, compute input): `self` in the formula, aclTensor on the device. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with that of **target**. The shape supports a maximum of eight dimensions. The data type must meet the [type deduction rules](../../../docs/en/context/deduction_relationship.md). [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND, NCL, NCHW, or NHWC.

    - <term>Atlas training series products</term>: The data type can be FLOAT16 or FLOAT32.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be BFLOAT16, FLOAT16, or FLOAT32.

  - **target** (aclTensor*, compute input): `target` in the formula, aclTensor on the device. The shape must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with that of **self**. The shape supports a maximum of eight dimensions. The data type must meet the [type deduction rules](../../../docs/en/context/deduction_relationship.md). [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND, NCL, NCHW, or NHWC.

    - <term>Atlas training series products</term>: The data type can be FLOAT16 or FLOAT32.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be BFLOAT16, FLOAT16, or FLOAT32.

  - **reduction** (int64_t, compute input): input to be applied to the reduction formula of the output, `reduction` in the formula, an integer on the host. The value can be 0 ('none') | 1 ('mean') | 2 ('sum'). **none** indicates that no reduction is applied; **mean** indicates that the sum of the output will be divided by the number of elements in the output; **sum** indicates that the output will be summed.

  - **beta** (float, compute input): The data type can be FLOAT. The value must be non-negative.

  - **result** (aclTensor*, compute output): loss function $\ell$ output in the formula. When `reduction` is `none`, the shape is the same as the broadcast result of **self** and **target**. When `reduction` is `mean` or `sum`, the shape is [ ]. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND, NCL, NCHW, or NHWC.

    - <term>Atlas training series products</term>: The data type can be FLOAT16 or FLOAT32.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be BFLOAT16, FLOAT16, or FLOAT32.

  - **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.

  - **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.


- **Returns:**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): The passed self, target, or result is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, target, or result is not supported.
                                    2. The shape of self, target, or result does not comply with the constraints.
                                    3. reduction does not comply with the constraints.
                                    4. beta does not comply with the constraints.
                                    5. The data types of self and target do not meet the type deduction rules.
                                    6. The shapes of self and target do not meet the broadcast relationship.
                                    7. When reduction is 0, the shape of result is different from the broadcast shapes of self and target.
  ```
## aclnnSmoothL1Loss

- **Parameters:**

  - **workspace** (void*, input): address of the workspace to be allocated on the device.

  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnSmoothL1LossGetWorkspaceSize**.

  - **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - **stream** (aclrtStream, input): stream for executing the task.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic compute:
  - **aclnnSmoothL1Loss** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_smooth_l1_loss.h"

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
  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> selfShape = {2, 2, 7, 7};
  std::vector<int64_t> targetShape = {2, 2, 7, 7};
  std::vector<int64_t> resultShape = {2, 2, 7, 7};

  // Create a self aclTensor.
  std::vector<float> selfData(GetShapeSize(selfShape)* 2, 1);
  aclTensor* self = nullptr;
  void *selfDeviceAddr = nullptr;
  ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a target aclTensor.
  std::vector<float> targetData(GetShapeSize(targetShape)* 2, 1);
  aclTensor* target = nullptr;
  void *targetDeviceAddr = nullptr;
  ret = CreateAclTensor(targetData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT16, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a result aclTensor.
  std::vector<float> resultData(GetShapeSize(resultShape)* 2, 1);
  aclTensor* result = nullptr;
  void *resultDeviceAddr = nullptr;
  ret = CreateAclTensor(resultData, resultShape, &resultDeviceAddr, aclDataType::ACL_FLOAT16, &result);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSmoothL1Loss.
  int64_t reduction = 0;
  float beta = 1.0;
  ret = aclnnSmoothL1LossGetWorkspaceSize(self, target, reduction, beta, result, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSmoothL1LossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSmoothL1Loss.
  ret = aclnnSmoothL1Loss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSmoothL1Loss failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(resultShape);
  std::vector<float> resultOutData(size, 0);
  ret = aclrtMemcpy(resultOutData.data(), resultOutData.size() * sizeof(resultOutData[0]), resultDeviceAddr,
                    size * sizeof(resultOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultOutData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(result);
  // 7. Release device resources. Set the parameters based on the API definition.
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
