# aclnnEmbeddingDenseBackward

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

Operator function: implements the backward computation of [aclnnEmbedding](../../gather_v2/docs/aclnnEmbedding_en.md) and accumulates the gradient row corresponding to the same index `indices` to out.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnEmbeddingDenseBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnEmbeddingDenseBackward** is called to perform computation.

- `aclnnStatus aclnnEmbeddingDenseBackwardGetWorkspaceSize(const aclTensor *grad, const aclTensor *indices, uint64_t numWeights, uint64_t paddingIdx, bool scaleGradByFreq, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnEmbeddingDenseBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnEmbeddingDenseBackwardGetWorkspaceSize

- **Parameters**

  - grad (aclTensor*, compute input): original gradient of the data, aclTensor on the device. The shape supports 2 to 8 dimensions. The shape after axis combination is the same as that after axis combination of indices except the last axis.[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [Data Format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT16 or FLOAT.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>, and Ascend 950PR/Ascend 950DT: The data type can be BFLOAT16, FLOAT16, or FLOAT.
  - indices (aclTensor*, compute input): index value corresponding to the grad input. It is an aclTensor on the device. The value range is [0, numWeights). The number of dimensions ranges from 1 to 8.[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported. The [Data Format](../../../docs/en/context/data_format.md) supports ND. The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, or BOOL.
  - **numWeights** (uint64_t, compute input): size of the first axis of the output tensor.
  - **paddingIdx** (uint64_t, compute input): used to pad 0 to the **paddingIdx** row in the output tensor. If **paddingIdx** is a negative number, no processing is performed.
  - scaleGradByFreq (bool, input): whether to scale the gradient based on the frequency of word occurrence. If the value is true, the result is scaled by word frequency. If the value is false, no processing is performed.
  - out (aclTensor*, compute output): output result of gradient summation, aclTensor on the device. The shape has 2 dimensions. The size of the first axis is numWeights, and the size of the last axis is the same as that of the last axis of grad. The data type must be the same as that of grad. The [Data Format](../../../docs/en/context/data_format.md) can be ND.
    - <term>Atlas training products</term>: The data type can be FLOAT16 or FLOAT.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>, and Ascend 950PR/Ascend 950DT: The data type can be BFLOAT16, FLOAT16, or FLOAT.
  - `workspaceSize` (uint64_t *, output): size of the workspace to be allocated on the device.
  - `executor` (aclOpExecutor **, output): operator executor, containing the operator computation process.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  ```text
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed grad, indices, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of grad, indices, or out is not supported.
                                      2. The shape of grad or indices exceeds 8D.
                                      3. The shapes of grad and indices do not meet the constraints.
                                      4. The shape of out does not comply with the inference result.
  ```

## aclnnEmbeddingDenseBackward

- **Parameters**

  * `workspace` (void*, input): address of the workspace to be allocated on the device.
  * **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling **aclnnEmbeddingDenseBackwardGetWorkspaceSize**.
  * `executor` (aclOpExecutor *, input): operator executor, containing the operator computation process.
  * `stream` (aclrtStream, input): stream for executing the task.
  
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- <term>Atlas training products</term>:
  - If **scale** is set to **true**, the last dimension of **grad** is defined as **embeddingDim**. An error is reported when its size exceeds the specified range. The valid ranges are as follows:
    - When **indices** is INT32, the following condition must be satisfied:
    $$
    embeddingDim < \frac{180192 - countsSize * 4}{36}
    $$
    - When **indices** is INT64, the following condition must be satisfied:
    $$
    embeddingDim < \frac{180192 - countsSize * 8}{20}
    $$
    - The formula for **countsSize** is as follows, where **coreNum** indicates the number of AI processor cores:
    $$
    countsSize = numWeights / coreNum + numWeights \% coreNum
    $$
- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
  - When the parameter shape exceeds the following limits, high precision cannot be guaranteed. If deterministic computation is enabled, high performance cannot be guaranteed either.
    - After **grad** is collapsed to a 2D shape, the first dimension exceeds INT32_MAX (2147483647).
    - **numWeights** exceeds INT32_MAX (2147483647).
  - When the collapsed dimension of **indices** exceeds INT32_INF (2139095040), high performance cannot be guaranteed.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_embedding_dense_backward.h"

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
  // (Boilerplate) Initialize resources.
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

  // Calculate the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID (deviceId) based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on API definitions.
  uint64_t numWeights = 4;
  uint64_t paddingIdx = 0;
  bool scaleGradByFreq = false;
  std::vector<int64_t> gradOutputShape = {2, 3};
  std::vector<int64_t> indicesShape = {2};
  std::vector<int64_t> outShape = {4, 3};
  void* gradOutputDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> gradOutputHostData = {1, 2, 3, 4, 5, 6};
  std::vector<int64_t> indicesHostData = {1, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create a gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Modify the API as required.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnEmbeddingDenseBackward.
  ret = aclnnEmbeddingDenseBackwardGetWorkspaceSize(gradOutput, indices, numWeights, paddingIdx, scaleGradByFreq, out,
                                                    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingDenseBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnEmbeddingDenseBackward.
  ret = aclnnEmbeddingDenseBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingDenseBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("resultData[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(indices);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(indicesDeviceAddr);
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
