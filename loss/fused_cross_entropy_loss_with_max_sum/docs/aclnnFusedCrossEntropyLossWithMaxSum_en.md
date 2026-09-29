# aclnnFusedCrossEntropyLossWithMaxSum

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/fused_cross_entropy_loss_with_max_sum)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Function: This operator is a part of the cross entropy calculation module in the vocabulary parallel scenario, which solves the problems of GPU memory and computing efficiency in ultra-large vocabulary scenarios. Currently, this operator is used to calculate the loss and softMax results.
- Formula:

    $$
    lossOut = log(sum\_exp\_logits) - predicted\_logits
    $$

    $$
    softMaxOutOptional = exp(vocab\_parallel\_logits -logits\_max.unsqueeze(dim = -1)) \ sum\_exp\_logits.unsqueeze(dim = -1)
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnFusedCrossEntropyLossWithMaxSum** is called to perform computation.

```Cpp
aclnnStatus aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize(
    const aclTensor* logitsMax,
    const aclTensor* sumExpLogits,
    const aclTensor* predictedLogits,
    float            labelSmoothing, 
    const aclTensor* inputOptional,
    const aclTensor* weightOptional,
    const aclTensor* vocabParallelLogitsOptional,
    aclTensor*       lossOut,
    aclTensor*       softMaxOutOptional,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnFusedCrossEntropyLossWithMaxSum(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1447px"><colgroup>
  <col style="width: 267px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 250px">
  <col style="width: 125px">
  <col style="width: 115px">
  <col style="width: 125px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-consecutive Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">logitsMax (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Maximum value of each row after matmul computation, that is, logitsMax in the formula.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">sumExpLogits (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Result of the exponential function after the matmul computation result is subtracted from the maximum value of each row, that is, sumExpLogits in the formula.</td>
      <td class="tg-0pky">The shape is the same as that of logitsMax.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">predictedLogits (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Indicates the result of maskedTargetOut after the difference between the matmul calculation result and the maximum value in each row is calculated. It is the predictedLogits in the formula.</td>
      <td class="tg-0pky">The shape is the same as that of logitsMax.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">labelSmoothing (float) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Label smoothing coefficient, which is used to relieve overfitting.</td>
      <td class="tg-0pky">Currently, only 0 is supported.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">inputOptional (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Left matrix of the Matmul input.</td>
      <td class="tg-0pky">Currently, only null pointers are supported.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">weightOptional (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Right matrix and weight matrix of the Matmul input.</td>
      <td class="tg-0pky">Currently, only null pointers are supported.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">vocabParallelLogitsOptional (aclTensor*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Matmul computation result, which is vocabParallelLogits in the formula.</td>
      <td class="tg-0pky">The first dimension of the shape must be the same as that of logitsMax.</td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">lossOut (aclTensor*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Intermediate variable, which is the loss in the formula.</td>
      <td class="tg-0pky">The shape is the same as that of logitsMax.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">softMaxOutOptional (aclTensor*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Intermediate variable, which is the vocabParallelLogits in the formula.</td>
      <td class="tg-0pky">The shape is the same as that of vocabParallelLogitsOptional.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the size of the workspace to be allocated on the device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the operator executor, including the operator computation process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.
  
  <table class="tg" style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 263px">
  <col style="width: 88px">
  <col style="width: 619px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Return Value</th>
      <th class="tg-0pky">Error Code</th>
      <th class="tg-0pky">Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">logitsMax, sumExpLogits, predictedLogits, and lossOut are null pointers.</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="4">161002</td>
      <td class="tg-0pky">The input and output data types are not supported.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The input and output shapes do not meet the constraints.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The input and output dimensions do not meet the constraints.</td>
    </tr>
    <tr>
      <td class="tg-0lax">labelSmoothing is not equal to 0.</td>
    </tr>
  </tbody>
  </table>

## aclnnFusedCrossEntropyLossWithMaxSum

- **Parameters:**

   <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
          <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize API.</td>
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
 
- **Returns**
  
  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - **aclnnFusedCrossEntropyLossWithMaxSum** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
