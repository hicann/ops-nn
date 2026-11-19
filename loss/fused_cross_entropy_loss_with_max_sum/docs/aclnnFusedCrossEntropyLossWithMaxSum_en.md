# aclnnFusedCrossEntropyLossWithMaxSum

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:20:18.754Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: This operator is part of the cross entropy computation module in vocabulary parallelism scenarios, addressing memory and computational efficiency issues for extremely large vocabularies. This part computes the loss and SoftMax results.
- Formula:

    $$
    lossOut = log(sum\_exp\_logits) - predicted\_logits
    $$

    $$
    softMaxOutOptional = exp(vocab\_parallel\_logits -logits\_max.unsqueeze(dim = -1)) \ sum\_exp\_logits.unsqueeze(dim = -1)
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnFusedCrossEntropyLossWithMaxSum" API to perform the computation.

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
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimensions (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">logitsMax (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Maximum value of each row after matmul computation, logitsMax in the formula.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">sumExpLogits (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">exp result of subtracting the maximum value of each row from the matmul computation result, sumExpLogits in the formula.</td>
      <td class="tg-0pky">The shape is consistent with logitsMax.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">predictedLogits (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Represents the result obtained by subtracting the row-wise maximum from the matmul computation result and filtering with maskedTargetOut, that is, predictedLogits in the formula.</td>
      <td class="tg-0pky">The shape is consistent with logitsMax.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">labelSmoothing (float)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Label smoothing coefficient, used to mitigate overfitting.</td>
      <td class="tg-0pky">Only 0 is currently supported.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">inputOptional(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Left matrix of the matmul input.</td>
      <td class="tg-0pky">Currently only supports null pointer input.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">weightOptional(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">matmul input right matrix, weight matrix.</td>
      <td class="tg-0pky">Currently only supports empty pointer as input.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">vocabParallelLogitsOptional (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">matmul computation result, vocabParallelLogits in the formula.</td>
      <td class="tg-0pky">The first dimension of shape must be consistent with the first dimension of logitsMax.</td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">lossOut (aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Intermediate variable, loss in the formula.</td>
      <td class="tg-0pky">shape is consistent with logitsMax.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">softMaxOutOptional (aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Intermediate variable, vocabParallelLogits in the formula.</td>
      <td class="tg-0pky">Shape is consistent with vocabParallelLogitsOptional; must be provided when vocabParallelLogitsOptional is not null, otherwise the first API returns 161001.</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size that needs to be applied on the Device side.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor, which contains the operator computation flow.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

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
      <td class="tg-0pky">logitsMax, sumExpLogits, predictedLogits, or lossOut is a null pointer; softMaxOutOptional is a null pointer while vocabParallelLogitsOptional is not null.</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="6">161002</td>
      <td class="tg-0pky">The data type of input and output is not within the supported range.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shape of input and output does not meet the constraints.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The dimensions of input and output do not meet the constraints.</td>
    </tr>
    <tr>
      <td class="tg-0pky">Dim 0 of vocabParallelLogitsOptional is inconsistent with that of logitsMax.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The vocab dimension (last dim) of vocabParallelLogitsOptional is 0.</td>
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
          <th>Parameter</th>
          <th>Input/Output</th>
          <th>Description</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>workspace</td>
          <td>Input</td>
          <td>Workspace memory address applied on the Device side.</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Input</td>
          <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize.</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>Input</td>
          <td>Op executor, which contains the operator computation flow.</td>
        </tr>
        <tr>
          <td>stream</td>
          <td>Input</td>
          <td>Stream for task execution.</td>
        </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnFusedCrossEntropyLossWithMaxSum defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  // Boilerplate, resource initialization
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
  // Call aclrtMalloc to apply for memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data on the host side to the memory on the device side.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. The construction depends on the API.
  std::vector<int64_t> logitsMaxShape = {2};
  std::vector<int64_t> sumExpLogitsShape = {2};
  std::vector<int64_t> predictedLogitsShape = {2};
  std::vector<int64_t> vocabParallelLogitsOptionalShape = {2, 2};
  std::vector<int64_t> lossOutShape = {2};
  std::vector<int64_t> softMaxOutOptionalShape = {2, 2};

  float labelSmoothing = 0;

  void* logitsMaxDeviceAddr = nullptr;
  void* sumExpLogitsDeviceAddr = nullptr;
  void* predictedLogitsDeviceAddr = nullptr;
  void* vocabParallelLogitsOptionalDeviceAddr = nullptr;
  void* lossOutDeviceAddr = nullptr;
  void* softMaxOutOptionalDeviceAddr = nullptr;

  aclTensor* logitsMax = nullptr;
  aclTensor* sumExpLogits = nullptr;
  aclTensor* predictedLogits = nullptr;
  // input and weight currently only support null pointers (nullptr), consistent with the parameter description.
  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* vocabParallelLogitsOptional = nullptr;
  aclTensor* lossOut = nullptr;
  aclTensor* softMaxOutOptional = nullptr;

  std::vector<float> logitsMaxHostData = {0.5, 1};
  std::vector<float> sumExpLogitsHostData = {0.5, 1};
  std::vector<float> predictedLogitsHostData = {0.5, 1};
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

  // input and weight currently only support null pointers (nullptr), so no Tensor is created.

  ret = CreateAclTensor(vocabParallelLogitsOptionalHostData, vocabParallelLogitsOptionalShape, &vocabParallelLogitsOptionalDeviceAddr, aclDataType::ACL_FLOAT, &vocabParallelLogitsOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(lossOutHostData, lossOutShape, &lossOutDeviceAddr, aclDataType::ACL_FLOAT, &lossOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(softMaxOutOptionalHostData, softMaxOutOptionalShape, &softMaxOutOptionalDeviceAddr, aclDataType::ACL_FLOAT, &softMaxOutOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call CANN operator library APIs. Replace with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnFusedCrossEntropyLossWithMaxSum.
  ret = aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize(logitsMax, sumExpLogits, predictedLogits, labelSmoothing, input, weight,
                                                          vocabParallelLogitsOptional, lossOut, softMaxOutOptional, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory of the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnFusedCrossEntropyLossWithMaxSum.
  ret = aclnnFusedCrossEntropyLossWithMaxSum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedCrossEntropyLossWithMaxSum failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify as required by the specific API definition.
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

  // 6. Release aclTensor and aclScalar. Modify as required by the specific API definition.
  aclDestroyTensor(logitsMax);
  aclDestroyTensor(sumExpLogits);
  aclDestroyTensor(predictedLogits);
  aclDestroyTensor(vocabParallelLogitsOptional);
  aclDestroyTensor(lossOut);
  aclDestroyTensor(softMaxOutOptional);

  // 7. Release device resources. Modify as required by the specific API definition.
  aclrtFree(logitsMaxDeviceAddr);
  aclrtFree(sumExpLogitsDeviceAddr);
  aclrtFree(predictedLogitsDeviceAddr);
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
