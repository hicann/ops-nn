# aclnnNLLLoss2dBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/nll_loss_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    √     |

## Function

Description: Performs backpropagation of the negative log-likelihood loss.

## Prototype

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnNLLLoss2dBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnNLLLoss2dBackward** is called to perform computation.

- `aclnnStatus aclnnNLLLoss2dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* weight, int64_t reduction, int64_t ignoreIndex, aclTensor* totalWeight, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnNLLLoss2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnNLLLoss2dBackwardGetWorkspaceSize

- **Parameters:**

  - **gradOutput** (aclTensor*, compute input): aclTensor on the device. The shape is three-dimensional (the first dimension is N) or one-dimensional (the number of elements is 1). [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas training series products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.

  - **self** (aclTensor*, compute input): aclTensor on the device. The shape is four-dimensional. The first dimension N indicates the batch size, and the second dimension C indicates the number of classes. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. Dimensions 0, 2, and 3 of **self** must be the same as dimensions 0, 1, and 2 of **target**, respectively. Otherwise, false is returned.
    - <term>Atlas training series products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.

  - **target** (aclTensor*, compute input): aclTensor on the device, indicating the real label. The shape is three-dimensional. The first dimension is N, the value range of each element is [0, C – 1]. The data type can be INT64 or UINT8. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.

  - **weight** (aclTensor*, compute input): aclTensor on the device, weight of each class. The shape is (C, ). [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas training series products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.

  - **reduction** (int64_t, compute input): int64_t on the host, specifying the reduction to be applied to the output. The value can be 0 ('none') | 1 ('mean') | 2 ('sum'). **none** indicates that no reduction is applied; **mean** indicates that the sum of the output will be divided by the number of elements in the output; **sum** indicates that the output will be summed. When **reduction** is **0**, the shapes of **target** and **gradOutput** must be the same. Otherwise, false is returned.

  - **ignoreIndex** (int64_t, compute input): int64_t on the host, target value that is ignored and does not affect the input gradient.

  - **totalWeight** (aclTensor*, compute input): aclTensor on the device. The data type is the same as that of **weight**, and the shape is (1,). When `reduction` is set to `mean`, `totalWeight` is the `weight` obtained from the corresponding position through `target`, with the `weight` corresponding to `ignoreIndex` removed and the remaining `weight` summed. When `reduction` is set to other values, this parameter is not processed by default. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas training series products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.

  - **out** (aclTensor*, compute output): aclTensor on the device. The shape is the same as that of **self**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. The data type is the same as that of **self**.
    - <term>Atlas training series products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.

  - **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.

  - **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

```
The first-phase API implements input parameter verification. The following errors may be thrown:
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed gradOutput, self, target, weight, totalWeight, or out is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of gradOutput, self, target, weight, totalWeight, or out is not supported.
                                  2. The data types of gradOutput, self, weight, totalWeight, and out are inconsistent.
                                  3. target is not a three-dimensional tensor, and self is not a four-dimensional tensor.
                                  4. The number of weight elements is not C.
                                  5. The numbers of elements in dimensions 0, 2, and 3 of self are not equivalent to the numbers of elements in dimensions 0, 1, and 2 of target.
                                  6. The number of totalWeight elements is not 1.
                                  7. When reduction is set to none, the number of dimensions of gradOutput is not 3, or the numbers of elements in dimensions 0, 1, and 2 of gradOutput are not equivalent to the numbers of elements in dimensions 0, 1, and 2 of target.
                                  8. When reduction is not set to none, the number of dimensions of gradOutput is greater than 1 or the number of elements is not 1.
                                  9. The reduction value is not within the range of 0 to 2.
```

## aclnnNLLLoss2dBackward

- **Parameters:**

  - **workspace** (void*, input): address of the workspace to be allocated on the device.

  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnNLLLoss2dBackwardGetWorkspaceSize**.

  - **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - **stream** (aclrtStream, input): stream for executing the task.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
    - **aclnnNLLLoss2dBackward** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_nll_loss2d_backward.h"

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
  std::vector<int64_t> gradShape = {3, 1, 1};
  std::vector<int64_t> selfShape = {3, 5, 1, 1};
  std::vector<int64_t> targetShape = {3, 1, 1};
  std::vector<int64_t> weightShape = {5};
  std::vector<int64_t> totalWeightShape = {1};
  std::vector<int64_t> outShape = {3, 5, 1, 1};

  void* gradDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* totalWeightDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* grad = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* totalWeight = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> gradHostData = {2.7, 2.6, 2.5};
  std::vector<float> selfHostData = {4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5};
  std::vector<int64_t> targetHostData = {2, 3, 1};
  std::vector<float> weightHostData = {1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<float> totalWeightHostData = {1.0};
  std::vector<float> outHostData = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int64_t reduction = 0;
  int64_t ignoreIndex = -100;

  // Create a grad aclTensor.
  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT64, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a totalWeight aclTensor.
  ret = CreateAclTensor(totalWeightHostData, totalWeightShape, &totalWeightDeviceAddr,
                        aclDataType::ACL_FLOAT, &totalWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnNLLLoss2dBackward.
  ret = aclnnNLLLoss2dBackwardGetWorkspaceSize(grad, self, target, weight, reduction, ignoreIndex, totalWeight, out,
                                               &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLoss2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnNLLLoss2dBackward.
  ret = aclnnNLLLoss2dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLoss2dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(grad);
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(weight);
  aclDestroyTensor(totalWeight);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(gradDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(totalWeightDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
