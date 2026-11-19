# aclnnApplyTopKTopP

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:06:30.712Z -->

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Performs top-k and top-p sampling filtering on the original input logits.

- Formula:
  - Sorts the input logits in ascending order along the last axis to obtain the corresponding sorted results sortedValue and sortedIndices.
  $$sortedValue, sortedIndices = sort(logits, dim=-1, descending=false, stable=true)$$
  - Compute the retention threshold (the k-th largest value).
  $$topKValue[b][v] = sortedValue[b][sortedValue.size(1) - k[b]]$$
  - Generate the mask for top-k filtering.
  $$topKMask = sortedValue < topKValue$$
  - Set values below the threshold to -Inf using topKMask.

  $$
  sortedValue[b][v] =
  \begin{cases}
  -Inf & \text{topKMask[b][v]=true}\\
  sortedValue[b][v] & \text{topKMask[b][v]=false}
  \end{cases}
  $$

  - Converts the top-k filtered data into a probability distribution along the last axis using softmax.
  $$probsValue = softmax(sortedValue, dim=-1)$$
  - Computes the cumulative probability along the last axis (accumulating from the smallest probability).
  $$probsSum = cumsum(probsValue, dim=-1)$$
  - Generate the top-p mask. Positions where the cumulative probability is less than or equal to 1-p need to be filtered out, and ensure that at least one element is retained per batch.
  $$topPMask[b][v] = probsSum[b][v] <= 1-p[b]$$
  $$topPMask[b][-1] = false$$
  - Set the values below the threshold to -Inf through topPMask.

  $$
  sortedValue[b][v] =
  \begin{cases}
  -Inf & \text{topPMask[b][v]=true}\\
  sortedValue[b][v] & \text{topPMask[b][v]=false}
  \end{cases}
  $$

  - Restore the filtered results to the original order based on sortedIndices.
  $out[b][v] = sortedValue[b][sortedIndices[b][v]]$
  where $0 \le b \lt logits.size(0), 0 \le v \lt logits.size(1)$.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnApplyTopKTopPGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnApplyTopKTopP" API to perform the computation.

```cpp
aclnnStatus aclnnApplyTopKTopPGetWorkspaceSize(
  const aclTensor* logits,
  const aclTensor* p,
  const aclTensor* k,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnApplyTopKTopP(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnApplyTopKTopPGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 359px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 143px">
  <col style="width: 146px">
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
      <td>logits</td>
      <td>Input</td>
      <td>Represents the data to be processed, namely logits in the formula.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>p</td>
      <td>Input</td>
      <td>Represents the top-p threshold, namely p in the formula.</td>
      <td>Value range: [0, 1].<br>The data type must be consistent with that of logits.<br>The shape must be consistent with logits.size(0).</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>k</td>
      <td>Input</td>
      <td>Specifies the top-k threshold, that is, k in the formula.</td>
      <td>If k is less than or equal to 0 or greater than or equal to logits.size(1), top-k filtering is not performed.<br>The shape must be consistent with logits.size(0).</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Filtered data, corresponding to out in the formula.</td>
      <td>The data type must be consistent with logits.<br>The shape must be consistent with logits.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>The passed-in logits or out is a null pointer, or both p and k are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of logits, p, k, or out is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of logits, p, or out do not match.</td>
    </tr>
    <tr>
      <td>The shapes of logits, p, k, or out do not match.</td>
    </tr>
  </tbody></table>

## aclnnApplyTopKTopP

- **Parameters:**
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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by calling the first-phase API aclnnApplyTopKTopPGetWorkspaceSize.</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnApplyTopKTopP defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_top_k_top_p.h"

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
  // Boilerplate code for resource initialization
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
  // Call aclrtMalloc to allocate memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. The construction must be custom based on the API.
  std::vector<int64_t> logitsShape = {3, 4};
  std::vector<int64_t> pShape = {3};
  std::vector<int64_t> kShape = {3};
  std::vector<int64_t> outShape = {3, 4};
  void* logitsDeviceAddr = nullptr;
  void* pDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* logits = nullptr;
  aclTensor* p = nullptr;
  aclTensor* k = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> logitsHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> pHostData = {0.2, 0.4, 0.6};
  std::vector<int32_t> kHostData = {1, 2, 3};
  std::vector<float> outHostData(12, 0);
  // Create the logits aclTensor.
  ret = CreateAclTensor(logitsHostData, logitsShape, &logitsDeviceAddr, aclDataType::ACL_FLOAT, &logits);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the p aclTensor.
  ret = CreateAclTensor(pHostData, pShape, &pDeviceAddr, aclDataType::ACL_FLOAT, &p);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the k aclTensor.
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_INT32, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnApplyTopKTopP.
  ret = aclnnApplyTopKTopPGetWorkspaceSize(logits, p, k, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyTopKTopPGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate Device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnApplyTopKTopP.
  ret = aclnnApplyTopKTopP(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyTopKTopP failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait synchronously for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the Device side to the Host side. Modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify it based on the specific API definition.
  aclDestroyTensor(logits);
  aclDestroyTensor(p);
  aclDestroyTensor(k);
  aclDestroyTensor(out);

  // 7. Release Device resources. Modify it based on the specific API definition.
  aclrtFree(logitsDeviceAddr);
  aclrtFree(pDeviceAddr);
  aclrtFree(kDeviceAddr);
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
