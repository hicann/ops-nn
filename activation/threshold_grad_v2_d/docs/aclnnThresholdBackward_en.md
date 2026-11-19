# aclnnThresholdBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T16:33:25.293Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Performs the backward of [aclnnThreshold](../../threshold_v2/docs/aclnnThreshold&aclnnInplaceThreshold.md).
- Formula:

  $$
  output =
  \begin{cases}
  gradOutput(i) & \text{if } self(i) > threshold \\
  0 & \text{otherwise}
  \end{cases}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnThresholdBackwardGetWorkspaceSize" to obtain input parameters and calculate the required workspace size based on the computation process. Then call "aclnnThresholdBackward" to perform the computation.

```Cpp
aclnnStatus aclnnThresholdBackwardGetWorkspaceSize(
  const aclTensor *gradOutput,
  const aclTensor *self,
  const aclScalar *threshold,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnThresholdBackward(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnThresholdBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1330px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 150px">
  <col style="width: 230px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
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
      <td>gradOutput (aclTensor*)</td>
      <td>Input</td>
      <td>gradOutput in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>dtype must be the same as that of self.</li><li>shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.</li></ul></td>
      <td>FLOAT, BFLOAT16, FLOAT16, INT32, INT8, UINT8, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>self in the formula.</td>
      <td>The data type and the data type of gradOutput must satisfy the data type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</td>
      <td>FLOAT, BFLOAT16, FLOAT16, INT32, INT8, UINT8, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>threshold (aclScalar*)</td>
      <td>Input</td>
      <td>The threshold in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The dtype must be the same as that of gradOutput.</li><li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with gradOutput.</li></ul></td>
      <td>FLOAT, BFLOAT16, FLOAT16, INT32, INT8, UINT8, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>

    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>out in the formula.</td>
      <td><ul><li>The dtype must be the same as that of self.</li><li>The shape must be the same as the shape of self and gradOutput after broadcast.</li></ul></td>
      <td>FLOAT, BFLOAT16, FLOAT16, INT32, INT8, UINT8, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied for on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The INT64 data type is not supported.
    <!-- end id7 -->
    <!-- npu="910,310p" id8 -->
    - <term>Atlas inference products</term> and <term>Atlas training products</term>: The BFLOAT16 and INT64 data types are not supported.
    <!-- end id8 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
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
      <td>The passed-in gradOutput or self is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of gradOutput or self is not supported.</td>
    </tr>
    <tr>
      <td>The shape of gradOutput or self exceeds 8 dimensions.</td>
    </tr>
    <tr>
      <td>The data types of gradOutput, out, and self are inconsistent.</td>
    </tr>
  </tbody></table>

## aclnnThresholdBackward

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnThresholdBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnThresholdBackward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_threshold_backward.h"

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
  // Boilerplate for resource initialization
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
  // Call aclrtMemcpy to copy host data to the device memory.
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
  // 1. (boilerplate) Device/stream initialization. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output as required by the API.
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* gradOutputDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* gradOutput = nullptr;
  aclScalar* threshold = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0.2, 1.2, 2.2, 3.2};
  std::vector<float> gradOutputHostData = {4.5, 4.4, 4.3, 4.2};
  std::vector<float> outHostData = {0.0, 0.0, 0.0, 0.0};
  float thresholdValue = 1.0f;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the threshold aclScalar.
  threshold = aclCreateScalar(&thresholdValue, aclDataType::ACL_FLOAT);
  CHECK_RET(threshold != nullptr, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnThresholdBackward.
  ret = aclnnThresholdBackwardGetWorkspaceSize(gradOutput, self, threshold, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThresholdBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnThresholdBackward.
  ret = aclnnThresholdBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThresholdBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Wait synchronously for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. Change it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Change it according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(gradOutput);
  aclDestroyScalar(threshold);
  aclDestroyTensor(out);

  // 7. Release device resources. Change it according to the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(gradOutputDeviceAddr);
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
