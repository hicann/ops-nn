# aclnnBinaryCrossEntropyWithLogitsTargetBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/activation/logsigmoid)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    ×     |

## Function

Computes logits of the input self, and performs backpropagation of the [BECLoss](../../sigmoid_cross_entropy_with_logits_v2/docs/aclnnBinaryCrossEntropyWithLogits_en.md) with respect to the target label value **target**.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnBinaryCrossEntropyWithLogitsTargetBackward** is called to perform computation.

  * `aclnnStatus aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, const aclTensor *weightOptional, const aclTensor *posWeightOptional, int64_t reduction, aclTensor *gradTarget, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnBinaryCrossEntropyWithLogitsTargetBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize

- **Parameters:**
  * **gradOutput** (aclTensor \*, compute input): gradient value of the previous step of network backpropagation, which is an aclTensor on the device. The data type can be FLOAT16, FLOAT, or BFLOAT16. The shape must be able to be [broadcast](../../../docs/en/context/broadcast_relationship.md) to the shape of self. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **self** (aclTensor \*, compute input): computation result of the previous layer in the forward propagation of the network, which is an aclTensor on the device. The data type can be FLOAT16, FLOAT, or BFLOAT16. The shape is less than or equal to 8D. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **target** (aclTensor \*, compute input): label value, which is an aclTensor on the device. The data type can be FLOAT, FLOAT16, or BFLOAT16. The shape must be the same as that of self, and must be less than or equal to 8D. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **weightOptional** (aclTensor \*, compute input): weight of binary cross entropy, which is an aclTensor on the device. The data type can be FLOAT16, FLOAT, or BFLOAT16. The shape must be able to be [broadcast](../../../docs/en/context/broadcast_relationship.md) to the shape of self. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) supports ND. If weightOptional is empty, a tensor filled with 1s is created based on the shape of self.
  * **posWeightOptional** (aclTensor \*, compute input): weight of the positive class, which is an aclTensor on the device. The data type can be FLOAT16, FLOAT, or BFLOAT16. The shape can be [broadcast](../../../docs/en/context/broadcast_relationship.md) to the shape of self. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) supports ND. If weightOptional is empty, a tensor filled with 1s is created based on the shape of self.
  * **reduction** (int64_t, compute input): reduction operation performed on the gradient computation result of binary cross entropy. It is an integer on the host. The data type can be INT64. Only **0**, **1**, and **2** are supported. **0** indicates that no operation is performed. **1** indicates that the average value of the results is used. **2** indicates that the results are summed up.
  * **gradTarget** (aclTensor\*, compute output): gradient computation result, which is an aclTensor on the device. The data type can be FLOAT, FLOAT16, or BFLOAT16. The data type must be the same as that of target. The shape must be the same as that of self, and must be less than or equal to 8D. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND and must be the same as that of self.
  * **workspaceSize** (uint64_t \*, output): size of the workspace to be allocated on the device.
  * **executor** (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed gradOutput, self, target, or gradTarget is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or data format of gradOutput, self, target, or gradTarget is not supported.
                                    2. When weightOptional and posWeightOptional are not null pointers, their data types or formats are not supported.
                                    3. The shapes of self, target, and gradTarget are inconsistent.
                                    4. When weightOptional and posWeightOptional are not null pointers, their shapes cannot be broadcast to the shape of self.
                                    5. The shape of gradOutput cannot be broadcast to the shape of self.
                                    6. The value of reduction is not 0, 1, or 2.
                                    7. The shapes of gradOutput, self, target, and gradTarget are greater than 8D.
  ```

## aclnnBinaryCrossEntropyWithLogitsTargetBackward

- **Parameters:**
  * **workspace** (void \*, input): address of the workspace to be allocated on the device.
  * **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API **aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize**.
  * **executor** (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  * **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic compute:
  - **aclnnBinaryCrossEntropyWithLogitsTargetBackward** defaults to a deterministic implementation.

## Example
The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_binary_cross_entropy_with_logits_target_backward.h"

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
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> targetShape = {2, 2};
  std::vector<int64_t> weightShape = {2, 2};
  std::vector<int64_t> posWeightShape = {2, 2};
  std::vector<int64_t> gradTargetShape = {2, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* posWeightDeviceAddr = nullptr;
  void* gradTargetDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* gradTarget = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* posWeight = nullptr;
  std::vector<float> gradOutputHostData = {0, 1, 2, 3};
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> targetHostData = {0.1, 0.1, 0.1, 0.1};
  std::vector<float> weightHostData = {0, 1, 2, 3};
  std::vector<float> posWeightHostData = {0, 1, 2, 3};
  std::vector<float> gradTargetHostData = {0, 0, 0, 0};
  int64_t reduction = 0;

  // Create a gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a posWeight aclTensor.
  ret = CreateAclTensor(posWeightHostData, posWeightShape, &posWeightDeviceAddr, aclDataType::ACL_FLOAT, &posWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gradTarget aclTensor.
  ret = CreateAclTensor(gradTargetHostData, gradTargetShape, &gradTargetDeviceAddr, aclDataType::ACL_FLOAT, &gradTarget);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBinaryCrossEntropyWithLogitsTargetBackward API call example
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  // Call the first-phase API of aclnnBinaryCrossEntropyWithLogitsTargetBackward.
  ret = aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize(gradOutput, self, target, weight, posWeight,
      reduction, gradTarget, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize failed. ERROR: %d\n",
                                          ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnBinaryCrossEntropyWithLogitsTargetBackward.
  ret = aclnnBinaryCrossEntropyWithLogitsTargetBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyWithLogitsTargetBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(gradTargetShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradTargetDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(weight);
  aclDestroyTensor(posWeight);
  aclDestroyTensor(gradTarget);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(posWeightDeviceAddr);
  aclrtFree(gradTargetDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
