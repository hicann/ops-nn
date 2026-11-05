# aclnnSyncBatchNormGatherStats

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:37:37.526Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Not Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not Supported
<!-- end id6 -->

## Function

- Description:
Gathers the mean and variance from all devices and updates the global mean and variance.

- Formula:

  $$
  batchMean = \frac{\sum^N_{i=0}{totalSum[i]}}{\sum^N_{i=0}{sampleCount[i]}}
  $$

  $$
  batchVar = \frac{\sum^N_{i=0}{totalSquareSum[i]}}{\sum^N_{i=0}{sampleCount[i]}} - batchMean^2
  $$

  $$
  batchInvstd = \frac{1}{\sqrt{batchVar + ε}}
  $$

  $$
  runningMean = runningMean*(1-momentum) + momentum*batchMean
  $$

  $$
  runningVar = runningVar*(1-momentum) + momentum*(batchVar*   \frac{\sum^N_{i=0}
  {sampleCount[i]}}{\sum^N_{i=0}{sampleCount[i]}-1})
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnSyncBatchNormGatherStatsGetWorkspaceSize" first to obtain the input parameters and calculate the required workspace size based on the computation flow, and then call "aclnnSyncBatchNormGatherStats" to perform the computation.

```Cpp
aclnnStatus aclnnSyncBatchNormGatherStatsGetWorkspaceSize(
  const aclTensor   *totalSum,
  const aclTensor   *totalSquareSum,
  const aclTensor   *sampleCount,
  aclTensor         *mean,
  aclTensor         *variance,
  float              momentum,
  float              eps,
  aclTensor         *batchMean,
  aclTensor         *batchInvstd,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnSyncBatchNormGatherStats(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnSyncBatchNormGatherStatsGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
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
      <td>totalSum (aclTensor*)</td>
      <td>Input</td>
      <td>Sum of channel features across devices, corresponding to totalSum in the formula.</td>
      <td>The first dimension must be greater than 0.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>totalSquareSum (aclTensor*)</td>
      <td>Input</td>
      <td>Square of channel features across devices, corresponding to totalSquareSum in the formula.</td>
      <td><ul><li>The first dimension must be greater than 0.</li><li>The shape and data type must be consistent with those of totalSum.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sampleCount (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the sample count of each device, corresponding to sampleCount in the formula.</td>
      <td><ul><li>The first dimension must be greater than 0.</li><li>The shape is consistent with the first dimension of totalSum.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean (aclTensor*)</td>
      <td>Input</td>
      <td>Mean value during the computation, corresponding to runningMean in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape is consistent with the second dimension of totalSum.</li><li>The data type is consistent with the data type of totalSum.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>variance (aclTensor*)</td>
      <td>Input</td>
      <td>Variance used in the computation, corresponding to runningVar in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is consistent with the second dimension of totalSum.</li><li>The data type is consistent with the data type of totalSum.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>momentum (float)</td>
      <td>Input</td>
      <td>Exponential smoothing parameters of runningMean and runningVar.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps (float)</td>
      <td>Input</td>
      <td>Offset used to prevent division by zero.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchMean (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the global batch mean, which is batchMean in the corresponding formula.</td>
      <td><ul><li>The shape is consistent with the second dimension of totalSum.</li><li>The data type is consistent with the data type of batchInvstd.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batchInvstd (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the reciprocal of the standard deviation, that is, batchInvstd in the corresponding formula.</td>
      <td><ul><li>The shape is consistent with the second dimension of totalSum.</li><li>The data type is consistent with that of batchMean.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    </tbody>
    </table>

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: The data type of the parameters `totalSum`, `totalSquareSum`, `sampleCount`, `mean`, `variance`, `batchMean`, and `batchInvstd` does not support BFLOAT16.
  <!-- end id7 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>The input totalSum, totalSquareSum, sampleCount, mean, variance, batchMean, and batchInvstd are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of the input totalSum, totalSquareSum, sampleCount, mean, variance, batchMean, and batchInvstd is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data type of the input totalSum, totalSquareSum, mean, variance, batchMean, and batchInvstd does not meet the constraints.</td>
    </tr>
    <tr>
      <td>The shape of the input totalSum, totalSquareSum, sampleCount, mean, variance, batchMean, and batchInvstd is not within the supported range.</td>
    </tr>
  </tbody></table>

## aclnnSyncBatchNormGatherStats

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnSyncBatchNormGatherStatsGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
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
  - aclnnSyncBatchNormGatherStats defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sync_batch_norm_gather_stats.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
      aclrtResetDevice(deviceId); aclFinalize(); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
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
  // 1. (boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> totalSumShape = {1, 2};
  std::vector<int64_t> totalSquareSumShape = {1, 2};
  std::vector<int64_t> sampleCountShape = {1};
  std::vector<int64_t> meanShape = {2};
  std::vector<int64_t> varShape = {2};
  std::vector<int64_t> batchMeanShape = {2};
  std::vector<int64_t> batchInvstdShape = {2};
  void* totalSumDeviceAddr = nullptr;
  void* totalSquareSumDeviceAddr = nullptr;
  void* sampleCountDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* varDeviceAddr = nullptr;
  void* batchMeanDeviceAddr = nullptr;
  void* batchInvstdDeviceAddr = nullptr;
  aclTensor* totalSum = nullptr;
  aclTensor* totalSquareSum = nullptr;
  aclTensor* sampleCount = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* var = nullptr;
  aclTensor* batchMean = nullptr;
  aclTensor* batchInvstd = nullptr;
  std::vector<float> totalSumData = {300, 400};
  std::vector<float> totalSquareSumData = {300, 400};
  std::vector<int32_t> sampleCountData = {400};
  std::vector<float> meanData = {400, 400};
  std::vector<float> varData = {400, 400};
  std::vector<float> batchMeanData = {0, 0};
  std::vector<float> batchInvstdData = {0, 0};
  float momentum = 1e-1;
  float eps = 1e-5;
  // Create the input totalSum aclTensor.
  ret = CreateAclTensor(totalSumData, totalSumShape, &totalSumDeviceAddr, aclDataType::ACL_FLOAT, &totalSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the input totalSquareSum aclTensor.
  ret = CreateAclTensor(totalSquareSumData, totalSquareSumShape, &totalSquareSumDeviceAddr, aclDataType::ACL_FLOAT, &totalSquareSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the input sampleCount aclTensor.
  ret = CreateAclTensor(sampleCountData, sampleCountShape, &sampleCountDeviceAddr, aclDataType::ACL_INT32, &sampleCount);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the input meanData aclTensor.
  ret = CreateAclTensor(meanData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the input varData aclTensor.
  ret = CreateAclTensor(varData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the input batchMeanData aclTensor.
  ret = CreateAclTensor(batchMeanData, batchMeanShape, &batchMeanDeviceAddr, aclDataType::ACL_FLOAT, &batchMean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the input batchInvstdData aclTensor.
  ret = CreateAclTensor(batchInvstdData, batchInvstdShape, &batchInvstdDeviceAddr, aclDataType::ACL_FLOAT, &batchInvstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnSyncBatchNormGatherStats.
  ret = aclnnSyncBatchNormGatherStatsGetWorkspaceSize(totalSum, totalSquareSum, sampleCount, mean, var, momentum, eps, batchMean, batchInvstd, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSyncBatchNormGatherStatsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnSyncBatchNormGatherStats.
  ret = aclnnSyncBatchNormGatherStats(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSyncBatchNormGatherStats failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the results from the device to the host. Modify it according to the specific API definition.
  auto size = GetShapeSize(batchMeanShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), batchMeanDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify it based on the specific API definition.
  aclDestroyTensor(totalSum);
  aclDestroyTensor(totalSquareSum);
  aclDestroyTensor(sampleCount);
  aclDestroyTensor(mean);
  aclDestroyTensor(var);
  aclDestroyTensor(batchMean);
  aclDestroyTensor(batchInvstd);
  // 7. Release device resources. Modify them based on the specific API definition.
  aclrtFree(totalSumDeviceAddr);
  aclrtFree(totalSquareSumDeviceAddr);
  aclrtFree(sampleCountDeviceAddr);
  aclrtFree(meanDeviceAddr);
  aclrtFree(varDeviceAddr);
  aclrtFree(batchMeanDeviceAddr);
  aclrtFree(batchInvstdDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
