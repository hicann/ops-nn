# aclnnBatchNormGatherStatsWithCounts

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/sync_batch_norm_gather_stats_with_counts)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     √    |

## Function

- Description:
  
  Collects the mean value and variance of data on all devices and updates the global mean value and the reciprocal of the standard deviation. The performance of **BatchNorm** is related to **BatchSize**. The larger the value of **BatchSize** is, the more accurate the statistics of **BatchNorm** will be. However, tasks such as detection occupy a large amount of Video RAM. A graphics card usually uses only a small number of images for training, for example, two images. As a result, the BatchNorm performance deteriorates. One solution is **SyncBatchNorm**, that is, all devices share the same **BatchNorm** to obtain the global statistics.

  During **aclnnBatchNormGatherStatsWithCounts** computation, [aclnnBatchNormStats] is used to compute the mean value and reciprocal of the standard deviation of a single device.
  <!--During aclnnBatchNormGatherStatsWithCounts computation, [aclnnBatchNormStats](./aclnnBatchNormStats_en.md) is used to compute the mean value and reciprocal of the standard deviation of a single device.-->

- Formula:

  $$
  y = \frac{(x-E[x])}{\sqrt{Var(x)+ eps}} * γ + β
  $$
  
  Where, the updated formulas for **runningMean** and **runningVar** are as follows:
  
  $$
      runningMean=runningMean*(1-momentum) + E[x]*momentum
  $$
  
  $$
      runningVar=runningVar*(1-momentum) + E[x]*momentum
  $$
  
## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnBatchNormGatherStatsWithCounts** is called to perform computation.

```Cpp
aclnnStatus aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize(
  const aclTensor* input,
  const aclTensor* mean,
  const aclTensor* invstd,
  aclTensor*       runningMean,
  aclTensor*       runningVar,
  double           momentum,
  double           eps,
  const aclTensor* counts,
  aclTensor*       meanAll,
  aclTensor*       invstdAll,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnBatchNormGatherStatsWithCounts(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize

- **Parameters:**

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
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Precaution</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>Input</td>
      <td>Sample value for statistics. It corresponds to `x` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The second dimension is fixed to the channel axis.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Mean value of the input data, corresponding to `E(x)` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The size of the first dimension must be the same as that of `invstd` and `counts`, and the size of the second dimension must be the same as that of the channel axis of `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invstd</td>
      <td>Input</td>
      <td>Reciprocal of the standard deviation of the input data, corresponding to the reciprocal of the square root of `Var(x)+ eps`.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The element values must be greater than 0. If the element values are less than or equal to 0, the precision is not guaranteed. </li><li>The size of the first dimension must be the same as that of `mean` and `counts`, and the size of the second dimension must be the same as that of the channel axis of `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningMean</td>
      <td>Input</td>
      <td>Optional. Mean value during the entire training process, corresponding to `runningMean` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The element values must be greater than 0. If the element values are less than or equal to 0, the precision is not guaranteed. </li><li>When `runningMean` is not a null pointer, the shape can be 1D and the size must be the same as that of the channel axis of `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningVar</td>
      <td>Input</td>
      <td>Optional. Variance during the entire training process, corresponding to `runningVar` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The element values must be greater than 0. If the element values are less than or equal to 0, the precision is not guaranteed. </li><li>When `runningVar` is not a null pointer, the shape can be 1D and the size must be the same as that of the channel axis of `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>momentum</td>
      <td>Input</td>
      <td>Exponential smoothing parameter of runningMean and runningVar, corresponding to `momentum` in the formula.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Input</td>
      <td>Value to be added to the variance to avoid division by zero. It corresponds to `eps` in the formula.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>counts</td>
      <td>Input</td>
      <td>Number of input data elements.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The element values can only be positive integers. In other scenarios, no guarantee is provided. </li><li>The size of the first dimension must be the same as that of mean and invstd.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>meanAll</td>
      <td>Output</td>
      <td>Mean value of all device data, corresponding to `E(x)` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape must be the same as the channel axis of the input parameter `input`. </li><li>The data format must be the same as that of `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invstdAll</td>
      <td>Output</td>
      <td>Reciprocal of the standard deviation of all device data, corresponding to the reciprocal of the square root of `Var(x)+ eps`.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape must be the same as the channel axis of the input parameter `input`. </li><li>The data format must be the same as that of `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>      
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
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
  </tbody>
  </table>

  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: The data types of `input`, `mean`, `invstd`, `runningMean`, `runningVar`, `counts`, `meanAll`, and `invstdAll` cannot be BFLOAT16.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown.

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
      <td>The passed input, mean, invstd, counts, meanAll, or invstdAll is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>The data type of input, meanAll, invstdAll, mean, invstd, runningMean, runningVar, or counts is not supported.</td>
    </tr>
    <tr>
      <td>The data format of input, mean, invstd, or counts is not supported.</td>
    </tr>
    <tr>
      <td>The shape of input is greater than 8D.</td>
    </tr>
    <tr>
      <td>The input has fewer than 2 dimensions.</td>
    </tr>
    <tr>
      <td>The shape of mean or invstd is not 2D.</td>
    </tr>
    <tr>
      <td>The sizes of counts, mean, and invstd in the first dimension are inconsistent.</td>
    </tr>
    <tr>
      <td>The size of mean or invstd in the second dimension is inconsistent with that of the channel axis of input.</td>
    </tr>
    <tr>
      <td>When runningMean is not null, the size of runningMean is inconsistent with that of the channel axis of input.</td>
    </tr>
    <tr>
      <td>When runningVar is not null, the size of runningVar is inconsistent with that of the channel axis of input.</td>
    </tr>
    <tr>
      <td>The shape of counts is not 1D.</td>
    </tr>
    <tr>
      <td>The shape of meanAll or invstdAll is inconsistent with that of the channel axis of input.</td>
    </tr>
  </tbody></table>

## aclnnBatchNormGatherStatsWithCounts

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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API **aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize**.</td>
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
  - **aclnnBatchNormGatherStatsWithCounts** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm_gather_stats_with_counts.h"

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
  std::vector<int64_t> inputShape = {2, 4, 2};
  std::vector<int64_t> meanShape = {4, 4};
  std::vector<int64_t> rMeanShape = {4};
  std::vector<int64_t> rVarShape = {4};
  std::vector<int64_t> countsShape = {4};
  std::vector<int64_t> invstdShape = {4, 4};
  std::vector<int64_t> meanAllShape = {4};
  std::vector<int64_t> invstdAllShape = {4};
  void* inputDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* rMeanDeviceAddr = nullptr;
  void* rVarDeviceAddr = nullptr;
  void* countsDeviceAddr = nullptr;
  void* invstdDeviceAddr = nullptr;
  void* meanAllDeviceAddr = nullptr;
  void* invstdAllDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* rMean = nullptr;
  aclTensor* rVar = nullptr;
  aclTensor* counts = nullptr;
  aclTensor* invstd = nullptr;
  aclTensor* meanAll = nullptr;
  aclTensor* invstdAll = nullptr;
  std::vector<float> inputHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> meanHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> rMeanHostData = {1, 2, 3, 4};
  std::vector<float> rVarHostData = {5, 6, 7, 8};
  std::vector<float> countsHostData = {1, 2, 3, 4};
  std::vector<float> invstdHostData = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  std::vector<float> meanAllHostData = {4, 0};
  std::vector<float> invstdAllHostData = {4, 0};
  double momentum = 1e-2;
  double eps = 1e-4;

  // Create an input aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a mean aclTensor.
  ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an rMean aclTensor.
  ret = CreateAclTensor(rMeanHostData, rMeanShape, &rMeanDeviceAddr, aclDataType::ACL_FLOAT, &rMean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an rVar aclTensor.
  ret = CreateAclTensor(rVarHostData, rVarShape, &rVarDeviceAddr, aclDataType::ACL_FLOAT, &rVar);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a counts aclTensor.
  ret = CreateAclTensor(countsHostData, countsShape, &countsDeviceAddr, aclDataType::ACL_FLOAT, &counts);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an invstd aclTensor.
  ret = CreateAclTensor(invstdHostData, invstdShape, &invstdDeviceAddr, aclDataType::ACL_FLOAT, &invstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a meanAll aclTensor.
  ret = CreateAclTensor(meanAllHostData, meanAllShape, &meanAllDeviceAddr, aclDataType::ACL_FLOAT, &meanAll);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  //Create an invstdAll aclTensor.
  ret = CreateAclTensor(invstdAllHostData, invstdAllShape, &invstdAllDeviceAddr, aclDataType::ACL_FLOAT, &invstdAll);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBatchNormGatherStatsWithCounts API call example
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  // Call the first-phase API of aclnnBatchNormGatherStatsWithCounts.
  ret = aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize(input, mean, invstd, rMean, rVar, momentum, eps, counts, meanAll, invstdAll, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnBatchNormGatherStatsWithCounts.
  ret = aclnnBatchNormGatherStatsWithCounts(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormGatherStatsWithCounts failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(meanAllShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), meanAllDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(mean);
  aclDestroyTensor(rMean);
  aclDestroyTensor(rVar);
  aclDestroyTensor(counts);
  aclDestroyTensor(invstd);
  aclDestroyTensor(meanAll);
  aclDestroyTensor(invstdAll);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(inputDeviceAddr);
  aclrtFree(meanDeviceAddr);
  aclrtFree(rMeanDeviceAddr);
  aclrtFree(rVarDeviceAddr);
  aclrtFree(countsDeviceAddr);
  aclrtFree(invstdDeviceAddr);
  aclrtFree(meanAllDeviceAddr);
  aclrtFree(invstdAllDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
