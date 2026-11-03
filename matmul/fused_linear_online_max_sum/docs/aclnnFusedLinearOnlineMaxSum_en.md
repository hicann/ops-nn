# aclnnFusedLinearOnlineMaxSum

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:02:54.304Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

- Description:

  Functionally equivalent to Megatron's implementation of matmul and fused\_vocab\_parallel\_cross\_entropy. Supports fusing matmul and celoss with device splitting along the vocabulary\_size dimension. Internally, the fusion is split by communication into [aclnnFusedLinearOnlineMaxSum](./aclnnFusedLinearOnlineMaxSum.md) and [aclnnFusedCrossEntropyLossWithMaxSum](../../../loss/fused_cross_entropy_loss_with_max_sum/docs/aclnnFusedCrossEntropyLossWithMaxSum.md), which need to be called sequentially to implement the complete function.

- Formula:

  1. Perform matrix multiplication of $input$ and $weight^T$ to obtain:

     $$
     vocabParallelLogitsOutOptional = input @ weight^T
     $$
  2. Calculate the maximum value of each row of $vocabParallelLogitsOutOptional$:

     $$
     logitsMaxLocalOut = max(vocabParallelLogitsOutOptional, dim=-1)
     $$
  3. Calculate the difference between $vocabParallelLogitsOutOptional$ and $logitsMaxLocalOut$:

     $$
     subRes[b][n] = vocabParallelLogitsOutOptional[b][n] - logitsMaxLocalOut[b]
     $$
  4. Calculate the sum of each row of $subRes$ after exponential operation.

     $$
     sumExpLogitsLocalOut = sum(exp(subRes), dim=-1)
     $$
  5. Compute the mask where $target$ is less than $vocabStartIndex$ or $target$ is greater than or equal to $vocabEndIndex$

     $$
     targetMask = (target < vocabStartIndex) | (target >= vocabEndIndex)
     $$
  6. Compute $maskedTargetOut$

     $$
     maskedTargetOut[b] =
     \begin{cases}
     0 & \text{targetMask[b]=true}\\
     target[b] - vocabStartIndex & \text{targetMask[b]=false}
     \end{cases}
     $$
  7. Compute $predictedLogitsLocalOut$

     $$
     predictedLogitsLocalOut[b] =
     \begin{cases}
     0 & \text{targetMask[b]=true}\\
     subRes[b][maskedTargetOut[b]] & \text{targetMask[b]=false}
     \end{cases}
     $$
  8. Compute $targetMaskOut$

     $$
     alignNum = (input.size(0) + 7) / 8 * 8\\
     maskBit[p] = \begin{cases}
     uint8(targetMask[p]) & \text{p < input.size(0)}\\
     1 & \text{input.size(0) <= p < alignNum}
     \end{cases} \\
     targetMaskOut[k] = 0b(maskBit[8*k:8*k+8])
     $$

  Where $0 \le b \lt input.size(0), 0 \le n \lt weight.size(0), 0 \le p \lt alignNum, 0 \le k \lt alignNum / 8$.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnFusedLinearOnlineMaxSumGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process. Then, call the "aclnnFusedLinearOnlineMaxSum" API to perform the computation.

```Cpp
aclnnStatus aclnnFusedLinearOnlineMaxSumGetWorkspaceSize(
  const aclTensor* input,
  const aclTensor* weight,
  const aclTensor* target,
  int64_t          vocabStartIndex,
  int64_t          vocabEndIndex,
  aclTensor*       logitsMaxLocalOut,
  aclTensor*       sumExpLogitsLocalOut,
  aclTensor*       predictedLogitsLocalOut,
  aclTensor*       targetMaskOut,
  aclTensor*       maskedTargetOut,
  aclTensor*       vocabParallelLogitsOutOptional,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnFusedLinearOnlineMaxSum(
  void*           workspace,
  uint64_t        workspaceSize,
  aclOpExecutor*  executor,
  aclrtStream     stream)
```

## aclnnFusedLinearOnlineMaxSumGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 220px">
  <col style="width: 120px">
  <col style="width: 240px">
  <col style="width: 330px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
   <tbody>
      <tr>
      <td>Input</td>
      <td>Input</td>
      <td>Indicates the left matrix of the matmul computation, which is input in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>input.size(1) must be less than or equal to 65534.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
     <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Indicates the right matrix of the matmul computation, which is weight in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type must be consistent with input.</li><li>weight.size(0) must be greater than 0.</li><li>weight.size(1) must be consistent with input.size(1).</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>target</td>
      <td>Input</td>
      <td>Indicates the target index, i.e., target in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>target.size(0) must be consistent with input.size(0).</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>vocabStartIndex</td>
      <td>Input</td>
      <td>Indicates the start index assigned to the current device, which is vocabStartIndex in the formula.</td>
      <td><ul><li>Value range: [0, max(target) - 1].</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>vocabEndIndex</td>
      <td>Input</td>
      <td>Indicates the end index (excluded from the shard) allocated to the current device, namely vocabEndIndex in the formula. The IDs handled by the current device form a left-closed and right-open interval [vocabStartIndex, vocabEndIndex).</td>
      <td><ul><li>The value range is [vocabStartIndex, vocabStartIndex + weight.size(0)], and must not exceed 16777216.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>logitsMaxLocalOut</td>
      <td>Output</td>
      <td>Indicates the maximum value of each row after the matmul computation, that is, logitsMaxLocalOut in the formula.</td>
      <td><ul><li>logitsMaxLocalOut.size(0) must be consistent with input.size(0).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>sumExpLogitsLocalOut</td>
      <td>Output</td>
      <td>Indicates the accumulated result within each row after exp computation is performed on the difference between the matmul computation result and the maximum value of each row, that is, sumExpLogitsLocalOut in the formula.</td>
      <td><ul><li>sumExpLogitsLocalOut.size(0) must be consistent with input.size(0).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>predictedLogitsLocalOut</td>
      <td>Output</td>
      <td>Indicates the result of subtracting the row-wise maximum value from the matmul computation result and then filtering by maskedTargetOut; that is, predictedLogitsLocalOut in the formula.</td>
      <td><ul><li>predictedLogitsLocalOut.size(0) must be consistent with input.size(0).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>targetMaskOut</td>
      <td>Output</td>
      <td>Indicates the mask used to filter the vocabulary, which is targetMaskOut in the formula.</td>
      <td><ul><li>The shape is [(input.size(0) + 7) / 8].</li></ul></td>
      <td>UINT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>maskedTargetOut</td>
      <td>Output</td>
      <td>Indicates the result of target filtered by targetMaskOut, which is maskedTargetOut in the formula.</td>
      <td><ul><li>The data type must be consistent with that of target.</li><li>maskedTargetOut.size(0) must be consistent with input.size(0).</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>vocabParallelLogitsOutOptional</td>
      <td>Output</td>
      <td>Indicates the matmul computation result. This is an optional output, corresponding to vocabParallelLogitsOutOptional in the formula.</td>
      <td><ul><li>The data type must be consistent with that of input.</li><li>The shape is [input.size(0), weight.size(0)].</li><li>When vocabParallelLogitsOutOptional is nullptr, the memory-saving branch is used; otherwise, the high-performance branch is used.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 267px">
  <col style="width: 103px">
  <col style="width: 604px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>input, weight, target, logitsMaxLocalOut, sumExpLogitsLocalOut, predictedLogitsLocalOut, targetMaskOut, or maskedTargetOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>The data types of input, weight, target, logitsMaxLocalOut, sumExpLogitsLocalOut, predictedLogitsLocalOut, targetMaskOut, and maskedTargetOut are not within the supported range.</td>
    </tr>
    <tr>
      <td>When vocabParallelLogitsOutOptional is not empty and its data type is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data formats of input, weight, target, logitsMaxLocalOut, sumExpLogitsLocalOut, predictedLogitsLocalOut, targetMaskOut, and maskedTargetOut are not within the supported range.</td>
    </tr>
    <tr>
      <td>When vocabParallelLogitsOutOptional is not empty and its data format is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of input, weight, target, logitsMaxLocalOut, sumExpLogitsLocalOut, predictedLogitsLocalOut, targetMaskOut, and maskedTargetOut do not meet the constraints.</td>
    </tr>
    <tr>
      <td>When vocabParallelLogitsOutOptional is not empty and its shape does not meet the constraints.</td>
    </tr>
    <tr>
      <td>The data types of input and weight are inconsistent.</td>
    </tr>
    <tr>
      <td>The data types of target and maskedTargetOut are inconsistent.</td>
    </tr>
    <tr>
      <td>vocabParallelLogitsOutOptional is not a null pointer and the data type of vocabParallelLogitsOutOptional is inconsistent with that of input.</td>
    </tr>
    <tr>
      <td>vocabStartIndex is less than 0.</td>
    </tr>
    <tr>
      <td>vocabEndIndex is less than vocabStartIndex.</td>
    </tr>
  </tbody></table>

## aclnnFusedLinearOnlineMaxSum

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnFusedLinearOnlineMaxSumGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Determinism:

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: aclnnFusedLinearOnlineMaxSum defaults to a deterministic implementation.

  <!-- end id7 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_fused_linear_online_max_sum.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to apply for memory on the device
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from the host to the device memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation
  // Set deviceId based on the actual device
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle check as needed
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. Customize the construction based on the API
  int64_t mSize = 128;
  int64_t kSize = 64;
  int64_t nSize = 256;
  std::vector<int64_t> inputShape = {mSize, kSize};
  std::vector<int64_t> weightShape = {nSize, kSize};
  std::vector<int64_t> targetShape = {mSize};
  std::vector<int64_t> logitsMaxLocalOutShape = {mSize};
  std::vector<int64_t> sumExpLogitsLocalOutShape = {mSize};
  std::vector<int64_t> predictedLogitsLocalOutShape = {mSize};
  std::vector<int64_t> targetMaskOutShape = {(mSize + 7) / 8};
  std::vector<int64_t> maskedTargetOutShape = {mSize};
  std::vector<int64_t> vocabParallelLogitsOutOptionalShape = {mSize, nSize};
  void* inputDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* logitsMaxLocalOutDeviceAddr = nullptr;
  void* sumExpLogitsLocalOutDeviceAddr = nullptr;
  void* predictedLogitsLocalOutDeviceAddr = nullptr;
  void* targetMaskOutDeviceAddr = nullptr;
  void* maskedTargetOutDeviceAddr = nullptr;
  void* vocabParallelLogitsOutOptionalDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* target = nullptr;
  aclTensor* logitsMaxLocalOut = nullptr;
  aclTensor* sumExpLogitsLocalOut = nullptr;
  aclTensor* predictedLogitsLocalOut = nullptr;
  aclTensor* targetMaskOut = nullptr;
  aclTensor* maskedTargetOut = nullptr;
  aclTensor* vocabParallelLogitsOutOptional = nullptr;
  std::vector<op::fp16_t> inputHostData(mSize * kSize, 1.0);
  std::vector<op::fp16_t> weightHostData(nSize * kSize, 1.0);
  std::vector<int32_t> targetHostData(mSize, 1);
  std::vector<float> logitsMaxLocalOutHostData(mSize, 0);
  std::vector<float> sumExpLogitsLocalOutHostData(mSize, 0);
  std::vector<float> predictedLogitsLocalOutHostData(mSize, 0);
  std::vector<uint8_t> targetMaskOutHostData((mSize + 7) / 8, 0);
  std::vector<int32_t> maskedTargetOutHostData(mSize, 0);
  std::vector<op::fp16_t> vocabParallelLogitsOutOptionalHostData(mSize * nSize, 0);
  int64_t vocabStartIndex = 0;
  int64_t vocabEndIndex = 64;
  // Create the input aclTensor
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT16, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the weight aclTensor
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT32, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the logitsMaxLocalOut aclTensor.
  ret = CreateAclTensor(logitsMaxLocalOutHostData, logitsMaxLocalOutShape, &logitsMaxLocalOutDeviceAddr, aclDataType::ACL_FLOAT, &logitsMaxLocalOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the sumExpLogitsLocalOut aclTensor.
  ret = CreateAclTensor(sumExpLogitsLocalOutHostData, sumExpLogitsLocalOutShape, &sumExpLogitsLocalOutDeviceAddr, aclDataType::ACL_FLOAT, &sumExpLogitsLocalOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the predictedLogitsLocalOut aclTensor.
  ret = CreateAclTensor(predictedLogitsLocalOutHostData, predictedLogitsLocalOutShape, &predictedLogitsLocalOutDeviceAddr, aclDataType::ACL_FLOAT, &predictedLogitsLocalOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the targetMaskOut aclTensor.
  ret = CreateAclTensor(targetMaskOutHostData, targetMaskOutShape, &targetMaskOutDeviceAddr, aclDataType::ACL_UINT8, &targetMaskOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the maskedTargetOut aclTensor.
  ret = CreateAclTensor(maskedTargetOutHostData, maskedTargetOutShape, &maskedTargetOutDeviceAddr, aclDataType::ACL_INT32, &maskedTargetOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the vocabParallelLogitsOutOptional aclTensor.
  ret = CreateAclTensor(vocabParallelLogitsOutOptionalHostData, vocabParallelLogitsOutOptionalShape, &vocabParallelLogitsOutOptionalDeviceAddr, aclDataType::ACL_FLOAT16, &vocabParallelLogitsOutOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnFusedLinearOnlineMaxSum.
  ret = aclnnFusedLinearOnlineMaxSumGetWorkspaceSize(input, weight, target, vocabStartIndex, vocabEndIndex, logitsMaxLocalOut, sumExpLogitsLocalOut, predictedLogitsLocalOut, targetMaskOut, maskedTargetOut, vocabParallelLogitsOutOptional, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearOnlineMaxSumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory using the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API aclnnFusedLinearOnlineMaxSum.
  ret = aclnnFusedLinearOnlineMaxSum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearOnlineMaxSum failed. ERROR: %d\n", ret); return ret);
  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code based on the specific API definition.
  auto size = GetShapeSize(logitsMaxLocalOutShape);
  std::vector<float> logitsMaxLocalOutResultData(size, 0);
  ret = aclrtMemcpy(logitsMaxLocalOutResultData.data(), logitsMaxLocalOutResultData.size() * sizeof(logitsMaxLocalOutResultData[0]), logitsMaxLocalOutDeviceAddr, size * sizeof(logitsMaxLocalOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("logitsMaxLocalOut[%ld] is: %f\n", i, logitsMaxLocalOutResultData[i]);
  }

  size = GetShapeSize(sumExpLogitsLocalOutShape);
  std::vector<float> sumExpLogitsLocalOutResultData(size, 0);
  ret = aclrtMemcpy(sumExpLogitsLocalOutResultData.data(), sumExpLogitsLocalOutResultData.size() * sizeof(sumExpLogitsLocalOutResultData[0]), sumExpLogitsLocalOutDeviceAddr, size * sizeof(sumExpLogitsLocalOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("sumExpLogitsLocalOut[%ld] is: %f\n", i, sumExpLogitsLocalOutResultData[i]);
  }

  size = GetShapeSize(predictedLogitsLocalOutShape);
  std::vector<float> predictedLogitsLocalOutResultData(size, 0);
  ret = aclrtMemcpy(predictedLogitsLocalOutResultData.data(), predictedLogitsLocalOutResultData.size() * sizeof(predictedLogitsLocalOutResultData[0]), predictedLogitsLocalOutDeviceAddr, size * sizeof(predictedLogitsLocalOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("predictedLogitsLocalOut[%ld] is: %f\n", i, predictedLogitsLocalOutResultData[i]);
  }

  size = GetShapeSize(targetMaskOutShape);
  std::vector<uint8_t> targetMaskOutResultData(size, 0);
  ret = aclrtMemcpy(targetMaskOutResultData.data(), targetMaskOutResultData.size() * sizeof(targetMaskOutResultData[0]), targetMaskOutDeviceAddr, size * sizeof(targetMaskOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("targetMaskOut[%ld] is: %hhu\n", i, targetMaskOutResultData[i]);
  }

  size = GetShapeSize(maskedTargetOutShape);
  std::vector<int32_t> maskedTargetOutResultData(size, 0);
  ret = aclrtMemcpy(maskedTargetOutResultData.data(), maskedTargetOutResultData.size() * sizeof(maskedTargetOutResultData[0]), maskedTargetOutDeviceAddr, size * sizeof(maskedTargetOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("maskedTargetOut[%ld] is: %d\n", i, maskedTargetOutResultData[i]);
  }

  size = GetShapeSize(vocabParallelLogitsOutOptionalShape);
  std::vector<op::fp16_t> vocabParallelLogitsOutOptionalResultData(size, 0);
  ret = aclrtMemcpy(vocabParallelLogitsOutOptionalResultData.data(), vocabParallelLogitsOutOptionalResultData.size() * sizeof(vocabParallelLogitsOutOptionalResultData[0]), vocabParallelLogitsOutOptionalDeviceAddr, size * sizeof(vocabParallelLogitsOutOptionalResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("vocabParallelLogitsOutOptional[%ld] is: %f\n", i, static_cast<float>(vocabParallelLogitsOutOptionalResultData[i]));
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(weight);
  aclDestroyTensor(target);
  aclDestroyTensor(logitsMaxLocalOut);
  aclDestroyTensor(sumExpLogitsLocalOut);
  aclDestroyTensor(predictedLogitsLocalOut);
  aclDestroyTensor(targetMaskOut);
  aclDestroyTensor(maskedTargetOut);
  aclDestroyTensor(vocabParallelLogitsOutOptional);

  // 7. Release device resources. Modify the code based on the specific API definition.
  aclrtFree(inputDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(logitsMaxLocalOutDeviceAddr);
  aclrtFree(sumExpLogitsLocalOutDeviceAddr);
  aclrtFree(predictedLogitsLocalOutDeviceAddr);
  aclrtFree(targetMaskOutDeviceAddr);
  aclrtFree(maskedTargetOutDeviceAddr);
  aclrtFree(vocabParallelLogitsOutOptionalDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
