# aclnnFusedCrossEntropyLossWithMaxSum

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/fused_cross_entropy_loss_with_max_sum)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    ×     |

## Function

- Description: This operator is a part of the cross entropy computation module in the vocabulary parallelism scenario. It solves the video memory and computing efficiency problems in the case of ultra-large vocabulary. This part involves the result of computing loss and softMax.
- Formula:

  $$ 
  lossOut = log(sum_exp_logits) - predicted_logits
  $$
  
  $$
  softMaxOutOptional = exp(vocab_parallel_logits -logits_max.unsqueeze(dim = -1)) \ sum_exp_logits.unsqueeze(dim = -1)
  $$
          

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnFusedCrossEntropyLossWithMaxSum** is called to perform computation.

- `aclnnStatus aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize(const aclTensor* logitsMax, const aclTensor* sumExpLogits, const aclTensor* predictedLogits, float labelSmoothing, const aclTensor* inputOptional, const aclTensor* weightOptional, const aclTensor* vocabParallelLogitsOptional, aclTensor* lossOut, aclTensor* softMaxOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor);`
- `aclnnStatus aclnnFusedCrossEntropyLossWithMaxSum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize

- **Parameters:**

  - **logitsMax** (aclTensor*, computation input): logitsMax in the formula, maximum value of each row after matmul computation, and aclTensor on the device. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. The shape can be 1D and the data type can be FLOAT.

  - **sumExpLogits** (aclTensor*, computation input): sumExpLogits in the formula, exp result obtained from the difference between the matmul computation result and the maximum value of each row, and aclTensor on the device. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. The shape can be 1D and is the same as that of **logitsMax**, and the data type can be FLOAT.
  
  - **predictedLogits** (aclTensor*, computation input): predictedLogits in the formula, result filtered by maskedTargetOut after the difference between the matmul computation result and the maximum value of each row is calculated, and aclTensor on the device. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. The shape can be 1D and is the same as that of **logitsMax**, and the data type can be FLOAT.

  - **labelSmoothing** (float, computation input): label smoothing coefficient, which is used to alleviate overfitting. Currently, only value **0** is supported.

  - **inputOptional** (aclTensor*, computation input): left matrix of the matmul input and aclTensor on the device. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. Currently, only null pointers are supported.
  
  - **weightOptional** (aclTensor*, computation input): weight matrix, right matrix of the matmul input, and aclTensor on the device. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. Currently, only null pointers are supported.

  - **vocabParallelLogitsOptional** (aclTensor*, computation input): vocabParallelLogits in the formula, matmul computation result, and aclTensor on the device. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. The shape can be 2D. The first dimension of the shape must be the same as that of **logitsMax**. The data type can be FLOAT16 or BFLOAT16.

  - **lossOut** (aclTensor*, computation output): loss in the formula, intermediate variable, and aclTensor on the device. The shape is the same as that of **logitsMax**. The [data format](../../../docs/en/context/data_formats.md) can be ND, and the data type can be FLOAT.

  - **softMaxOutOptional** (aclTensor*, computation output): vocabParallelLogits in the formula, intermediate variable, and aclTensor on the device. The shape is the same as that of **vocabParallelLogitsOptional**. The [data format](../../../docs/en/context/data_formats.md) can be ND, and the data type can be FLOAT.

  - **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.

  - **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

```
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001(ACLNN_ERR_PARAM_NULLPTR): 1. The self, index, or out is a null pointer.
  161002(ACLNN_ERR_PARAM_INVALID): 1. The data type of the input or output parameter is not supported.
                                   2. The dimensions of the input and output parameters are not supported.
                                   3. The shape of the input or output parameter does not meet the constraints.
                                   4. The value of labelSmoothing is not 0.
```

## aclnnFusedCrossEntropyLossWithMaxSum

- **Parameters:**

  - **workspace** (void*, input): address of the workspace to be allocated on the device.

  - **workspaceSize** (uint64_t, input): workspace size allocated on the device, which is obtained by the first-phase API **aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize**.

  - **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - **stream** (aclrtStream, input): stream for executing the task.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic computation:
  - **aclnnFusedCrossEntropyLossWithMaxSum** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_cross_entropy_loss_with_max_sum.h"

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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
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

  // 2. Construct the input and output based on the API definition.
  std::vector<int64_t> logitsMaxShape = {2};
  std::vector<int64_t> sumExpLogitsShape = {2};
  std::vector<int64_t> predictedLogitsShape = {2};
  std::vector<int64_t> inputShape = {2};
  std::vector<int64_t> weightShape = {2};
  std::vector<int64_t> vocabParallelLogitsOptionalShape = {2, 2};
  std::vector<int64_t> lossOutShape = {2};
  std::vector<int64_t> softMaxOutOptionalShape = {2, 2};

  float labelSmoothing = 0;

  void* logitsMaxDeviceAddr = nullptr;
  void* sumExpLogitsDeviceAddr = nullptr;
  void* predictedLogitsDeviceAddr = nullptr;
  void* inputDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* vocabParallelLogitsOptionalDeviceAddr = nullptr;
  void* lossOutDeviceAddr = nullptr;
  void* softMaxOutOptionalDeviceAddr = nullptr;

  aclTensor* logitsMax = nullptr;
  aclTensor* sumExpLogits = nullptr;
  aclTensor* predictedLogits = nullptr;
  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* vocabParallelLogitsOptional = nullptr;
  aclTensor* lossOut = nullptr;
  aclTensor* softMaxOutOptional = nullptr;

  std::vector<float> logitsMaxHostData = {0.5, 1};
  std::vector<float> sumExpLogitsHostData = {0.5, 1};
  std::vector<float> predictedLogitsHostData = {0.5, 1};
  std::vector<float> inputHostData = {0, 1};
  std::vector<float> weightHostData = {0, 1};
  std::vector<float> vocabParallelLogitsOptionalHostData = {1, 0.5, 0.5, 1};
  std::vector<float> lossOutHostData = {0, 0};
  std::vector<float> softMaxOutOptionalHostData = {0, 0, 0, 0};
  // Create an aclTensor.
  ret = CreateAclTensor(logitsMaxHostData, logitsMaxShape, &logitsMaxDeviceAddr, aclDataType::ACL_FLOAT, &logitsMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(sumExpLogitsHostData, sumExpLogitsShape, &sumExpLogitsDeviceAddr, aclDataType::ACL_FLOAT, &sumExpLogits);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(predictedLogitsHostData, predictedLogitsShape, &predictedLogitsDeviceAddr, aclDataType::ACL_FLOAT, &predictedLogits);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);


  ret = CreateAclTensor(vocabParallelLogitsOptionalHostData, vocabParallelLogitsOptionalShape, &vocabParallelLogitsOptionalDeviceAddr, aclDataType::ACL_FLOAT, &vocabParallelLogitsOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(lossOutHostData, lossOutShape, &lossOutDeviceAddr, aclDataType::ACL_FLOAT, &lossOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(softMaxOutOptionalHostData, softMaxOutOptionalShape, &softMaxOutOptionalDeviceAddr, aclDataType::ACL_FLOAT, &softMaxOutOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnFusedCrossEntropyLossWithMaxSum.
  ret = aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize(logitsMax, sumExpLogits, predictedLogits, labelSmoothing, input, weight, 
                                                          vocabParallelLogitsOptional, lossOut, softMaxOutOptional, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnFusedCrossEntropyLossWithMaxSum.
  ret = aclnnFusedCrossEntropyLossWithMaxSum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedCrossEntropyLossWithMaxSum failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(lossOutShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), lossOutDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  size = GetShapeSize(softMaxOutOptionalShape);
  std::vector<float> secondResultData(size, 0);
  ret = aclrtMemcpy(secondResultData.data(), secondResultData.size() * sizeof(secondResultData[0]), softMaxOutOptionalDeviceAddr,
                    size * sizeof(secondResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, secondResultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(logitsMax);
  aclDestroyTensor(sumExpLogits);
  aclDestroyTensor(predictedLogits);
  aclDestroyTensor(input);
  aclDestroyTensor(weight);
  aclDestroyTensor(vocabParallelLogitsOptional);
  aclDestroyTensor(lossOut);
  aclDestroyTensor(softMaxOutOptional);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(logitsMaxDeviceAddr);
  aclrtFree(sumExpLogitsDeviceAddr);
  aclrtFree(predictedLogitsDeviceAddr);
  aclrtFree(inputDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(vocabParallelLogitsOptionalDeviceAddr);
  aclrtFree(lossOutDeviceAddr);
  aclrtFree(softMaxOutOptionalDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
