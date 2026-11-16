# aclnnPreluBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T12:53:46.143Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

Description: Performs backward computation of [aclnnPrelu](../../p_relu/docs/aclnnPrelu.md).

The formula for gradInput is as follows:

$$
gradInput_{i,j,...}=
\begin{cases}
gradOutput_{i,j,...}, & if\ self_{i,j,...} > 0 \\
gradOutput_{i,j,...} * weight_{i}, & if\ self_{i,j,...} <= 0
\end{cases}
$$

The formula for gradWeight is as follows:

$$
gradWeight_{j}=\sum_{i,...}
\begin{cases}
0, & if\ self_{i,j,...} > 0 \\
gradOutput_{i,j,...} * self_{i,j,...}, & if\ self_{i,j,...} <= 0
\end{cases}
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnPreluBackwardGetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the operator computation process, and then call "aclnnPreluBackward" to perform computation.

```Cpp
aclnnStatus aclnnPreluBackwardGetWorkspaceSize(
  const aclTensor*  gradOutput,
  const aclTensor*  self,
  const aclTensor*  weight,
  aclTensor*        gradInput,
  aclTensor*        gradWeight,
  uint64_t*         workspaceSize,
  aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnPreluBackward(
  void*            workspace,
  uint64_t         workspace_size,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnPreluBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1490px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 240px">
  <col style="width: 300px">
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
      <td>Gradient value for backward propagation. gradOutput in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>dtype must be the same as that of self.</li><li>shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self, and the shape after broadcasting must be the same as the shape of self.</li></ul></td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Forward input value of prelu. self in the formula.</td>
      <td>Supports empty tensors.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight (aclTensor*)</td>
      <td>Input</td>
      <td>Weight of PReLU, which is the weight in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The dtype must be the same as that of self.</li><li>If the shape of self has more than 1 dimension, weight can have the same number of dimensions as self, with the second dimension having the same value as that of self and the other dimensions set to 1; alternatively, weight is a 1D tensor whose number of elements equals the value of the second dimension in the shape of self.</li><li>Otherwise, the number of elements in weight is 1.</li></ul></td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInput (aclTensor*)</td>
      <td>Output</td>
      <td>Gradient value of self.</td>
      <td><ul><li>dtype must be the same as self.</li><li>shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with gradOutput.</li><li>The shape and data type of gradInput are the same as those of self.</li></ul></td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeight (aclTensor*)</td>
      <td>Output</td>
      <td>Gradient value of weight.</td>
      <td><ul><li>The dtype must be the same as that of self.</li><li>The data type must be the same as that of weight.</li><li>The shape of gradWeight must be the same as the shape of weight.</li></ul></td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
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

   <!-- npu="910" id7 -->
   - <term>Atlas training products</term>: Data types FLOAT16 and FLOAT32 are supported.
   <!-- end id7 -->

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
      <td>The input gradOutput, self, weight, gradInput, or gradWeight is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type of gradOutput, self, weight, gradInput, or gradWeight is out of the supported range.</td>
    </tr>
    <tr>
      <td>The data types of gradOutput, self, weight, gradInput, and gradWeight are different.</td>
    </tr>
    <tr>
      <td>gradOutput, self, weight, gradInput, or gradWeight has more than 8 dimensions.</td>
    </tr>
    <tr>
      <td>The number of elements in weight does not equal the number of channels of self or 1.</td>
    </tr>
    <tr>
      <td>When weight has one element, the shape of gradWeight is not the same as that of weight.</td>
    </tr>
    <tr>
      <td>The shapes of gradOutput and self must satisfy the broadcast relationship, and their broadcasted shape must match the shape of self.</td>
    </tr>
  </tbody></table>

## aclnnPreluBackward

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
      <td>Memory address of the workspace requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device, obtained by calling the first-phase API aclnnPreluBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnPreluBackward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_prelu_backward.h"

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
  // Boilerplate: resource initialization
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
  // Call aclrtMalloc to apply for memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from the host to the device memory.
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
  // 1. (boilerplate) device/stream initialization, see the acl API manual.
  // Set deviceId based on the actual device in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output as required by the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> weightShape = {2};
  std::vector<int64_t> gradOutputShape = {4, 2};
  std::vector<int64_t> gradInputShape = {4, 2};
  std::vector<int64_t> gradWeightShape = {2};

  void* selfDeviceAddr = nullptr;
  void* gradOutputDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  void* gradWeightDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* gradInput = nullptr;
  aclTensor* gradWeight = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> weightHostData = {0.5, 0.5};
  std::vector<float> gradOutputHostData = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<float> gradInputHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> gradWeightHostData = {0, 0};

  // Create the weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT,
  &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the gradInput aclTensor.
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the gradWeight aclTensor.
  ret = CreateAclTensor(gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnPreluBackward.
  ret = aclnnPreluBackwardGetWorkspaceSize(gradOutput, self, weight, gradInput, gradWeight, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPreluBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnPreluBackward.
  ret = aclnnPreluBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPreluBackward failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output values. Copy the result from device memory to the host. Change the code based on the specific API definition.
  auto gradInputSize = GetShapeSize(gradInputShape);
  std::vector<float> gradInputResultData(gradInputSize, 0);
  ret = aclrtMemcpy(gradInputResultData.data(), gradInputResultData.size() * sizeof(gradInputResultData[0]), gradInputDeviceAddr, gradInputSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < gradInputSize; i++) {
    LOG_PRINT("gradInput[%ld] is: %f\n", i, gradInputResultData[i]);
  }

  auto gradWeightSize = GetShapeSize(gradWeightShape);
  std::vector<float> gradWeightResultData(gradWeightSize, 0);
  ret = aclrtMemcpy(gradWeightResultData.data(), gradWeightResultData.size() * sizeof(gradWeightResultData[0]), gradWeightDeviceAddr, gradWeightSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < gradWeightSize; i++) {
    LOG_PRINT("gradWeight[%ld] is: %f\n", i, gradWeightResultData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Change the code based on the specific API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(weight);
  aclDestroyTensor(gradInput);
  aclDestroyTensor(gradWeight);

  // 7. Release device resources. Modify the code based on the actual API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(gradInputDeviceAddr);
  aclrtFree(gradWeightDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
