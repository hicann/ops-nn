# aclnnSyncBatchNormGatherStats

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/sync_batch_norm_gather_stats)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description:
Collects the mean value and variance of data on all devices and updates the global mean value and variance.

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
     runningVar = runningVar*(1-momentum) + momentum*(batchVar* \frac{\sum^N_{i=0}
     {sampleCount[i]}}{\sum^N_{i=0}{sampleCount[i]}-1})
     $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnSyncBatchNormGatherStatsGetWorkspaceSize** is called to obtain the input parameters and to compute the workspace size required for computation. Then, **aclnnSyncBatchNormGatherStats** is called to perform computation.

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

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 190px">
  <col style="width: 190px">
  <col style="width: 145px">
  </colgroup>
  <table><thead>
  <tr>
    <th>Name</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Usage Notes</th>
    <th>Data Type</th>
    <th>Data Format</th>
    <th>Dimension (Shape)</th>
    <th>Non-contiguous Tensor</th>
  </tr></thead>
    <tbody>
    <tr>
        <td>totalSum</td>
        <td>Input</td>
        <td>Sum of channel features on each device, corresponding to totalSum in the formula.</td>
        <td>The first dimension must be greater than 0.</td>
        <td>BFLOAT16, FLOAT16, FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>totalSquareSum</td>
        <td>Input</td>
        <td>Square of channel features on each device, corresponding to totalSquareSum in the formula.</td>
        <td><ul><li>The first dimension must be greater than 0. </li><li>The shape is the same as that of totalSum.</li></ul></td>
        <td>BFLOAT16, FLOAT16, FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
    </tr>
    <tr>
        <td>sampleCount</td>
        <td>Input</td>
        <td>Number of samples on each device, corresponding to sampleCount in the formula.</td>
        <td><ul><li>The first dimension must be greater than 0. </li><li>The shape must be the same as the first dimension of totalSum.</li></ul></td>
        <td>BFLOAT16, FLOAT16, FLOAT, INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>mean</td>
        <td>Input</td>
        <td>Mean value during the compute process, corresponding to runningMean in the formula.</td>
        <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as the second dimension of totalSum.</li></ul></td>
        <td>BFLOAT16, FLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>variance</td>
        <td>Input</td>
        <td>Variance during the compute process, corresponding to runningVar in the formula.</td>
        <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as the second dimension of totalSum.</li></ul></td>
        <td>BFLOAT16, FLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>momentum</td>
        <td>Input</td>
        <td>Exponential smoothing parameter of runningMean and runningVar.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>eps</td>
        <td>Input</td>
        <td>Used to prevent the offset of dividing by 0.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>batchMean</td>
        <td>Output</td>
        <td>Global batch mean value, corresponding to batchMean in the formula.</td>
        <td><ul><li>The first dimension must be greater than 0.</li></ul></td>
        <td>BFLOAT16, FLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>batchInvstd</td>
        <td>Output</td>
        <td>Reciprocal of standard deviation, corresponding to batchInvstd in the formula.</td>
        <td><ul><li>The first dimension must be greater than 0.</li></ul></td>
        <td>BFLOAT16, FLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Size of the workspace required to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Operator executor, containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>
    </tbody>
    </table>

  - <term>Atlas A3 training series products/Atlas A3 inference series products</term> and <term>Atlas A2 training series products/Atlas A2 inference series products</term>: The data type of `totalSum`, `totalSquareSum`, `mean`, `variance`, `batchMean` and `batchInvstd` cannot be BFLOAT16.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:
  
  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed totalSum, totalSquareSum, sampleCount, mean, variance, batchMean, or batchInvstd is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type of the input totalSum, totalSquareSum, sampleCount, mean, variance, batchMean, or batchInvstd is not supported.</td>
    </tr>
    <tr>
      <td>The data format of the input totalSum, totalSquareSum, sampleCount, mean, or variance is not supported.</td>
    </tr>
    <tr>
      <td>The shape of the input totalSum, totalSquareSum, sampleCount, mean, variance, batchMean, or batchInvstd is not supported.</td>
    </tr>
  </tbody></table>

## aclnnSyncBatchNormGatherStats

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnSyncBatchNormGatherStatsGetWorkspaceSize.</td>
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

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
  - **aclnnSyncBatchNormGatherStats** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

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

  // 2. Construct the input and output based on the API definition.
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
  // Create an input totalSum aclTensor.
  ret = CreateAclTensor(totalSumData, totalSumShape, &totalSumDeviceAddr, aclDataType::ACL_FLOAT, &totalSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an input totalSquareSum aclTensor.
  ret = CreateAclTensor(totalSquareSumData, totalSquareSumShape, &totalSquareSumDeviceAddr, aclDataType::ACL_FLOAT, &totalSquareSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an input sampleCount aclTensor.
  ret = CreateAclTensor(sampleCountData, sampleCountShape, &sampleCountDeviceAddr, aclDataType::ACL_INT32, &sampleCount);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an input meanData aclTensor.
  ret = CreateAclTensor(meanData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an input varData aclTensor.
  ret = CreateAclTensor(varData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an input batchMeanData aclTensor.
  ret = CreateAclTensor(batchMeanData, batchMeanShape, &batchMeanDeviceAddr, aclDataType::ACL_FLOAT, &batchMean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an input batchInvstdData aclTensor.
  ret = CreateAclTensor(batchInvstdData, batchInvstdShape, &batchInvstdDeviceAddr, aclDataType::ACL_FLOAT, &batchInvstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnSyncBatchNormGatherStats.
  ret = aclnnSyncBatchNormGatherStatsGetWorkspaceSize(totalSum, totalSquareSum, sampleCount, mean, var, momentum, eps, batchMean, batchInvstd, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSyncBatchNormGatherStatsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnSyncBatchNormGatherStats.
  ret = aclnnSyncBatchNormGatherStats(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSyncBatchNormGatherStats failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(batchMeanShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), batchMeanDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(totalSum);
  aclDestroyTensor(totalSquareSum);
  aclDestroyTensor(sampleCount);
  aclDestroyTensor(mean);
  aclDestroyTensor(var);
  aclDestroyTensor(batchMean);
  aclDestroyTensor(batchInvstd);
  // 7. Release device resources. Modify the configuration based on the API definition.
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
