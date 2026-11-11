# aclnnForeachNonFiniteCheckAndUnscale

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T01:23:23.753Z -->

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

- Description: Traverses all tensors in scaledGrads to check for Inf or NaN. If found, sets foundInf to 1.0; otherwise, keeps foundInf unchanged. Unscales all tensors in scaledGrads.

- Formula:

  $$
  foundInf = \begin{cases}1.0, & if (Inf \in  scaledGrads) or (NaN \in scaledGrads),\\
    foundInf, &otherwise.
  \end{cases}
  $$

  $$
   scaledGrads_i = {scaledGrads}_{i}*{invScale}.
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the "aclnnForeachNonFiniteCheckAndUnscale" API to perform the computation.

```Cpp
aclnnStatus aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(
  const aclTensorList *scaledGrads,
  const aclTensor     *foundInf,
  const aclTensor     *invScale,
  uint64_t            *workspaceSize,
  aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnForeachNonFiniteCheckAndUnscale(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize

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
      <td>scaledGrads (aclTensorList*)</td>
      <td>input/output</td>
      <td>Input and output tensor list for the unscale computation, that is, the `scaledGrads` in the corresponding formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>All tensors in this parameter must have the same data type.</li><li>A maximum of 256 tensors is supported.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>foundInf (aclTensor*)</td>
      <td>input/output</td>
      <td>Tensor that marks whether the input scaledGrads contains Inf or NaN, that is, the `foundInf` in the corresponding formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Contains only one element.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>invScale (aclTensor*)</td>
      <td>Input</td>
      <td>Tensor for inverse scaling computation, corresponding to `invScale` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Contains only one element.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-8</td>
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
      <td>Returns the op executor that contains the operator computation flow.</td>
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
      <td>scaledGrads, foundInf, or invScale is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types of scaledGrads, foundInf, or invScale are not within the supported range.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="3">561002</td>
      <td>The length of scaledGrads exceeds the limit.</td>
    </tr>
    <tr>
      <td>The data types of tensors in scaledGrads are different.</td>
    </tr>
    <tr>
      <td>The number of elements in foundInf or invScale is not 1.</td>
    </tr>
  </tbody></table>

## aclnnForeachNonFiniteCheckAndUnscale

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
      <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnForeachNonFiniteCheckAndUnscale defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_non_finite_check_and_unscale.h"

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
  // Boilerplate code for resource initialization.
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
  // Call aclrtMemcpy to copy data from the host to the device memory.
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
  // 1. (boilerplate) device/stream initialization. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct input and output. Custom construction is required based on the API.
  std::vector<int64_t> selfShape1 = {2, 3};
  std::vector<int64_t> selfShape2 = {1, 3};
  std::vector<int64_t> foundInfShape = {1};
  std::vector<int64_t> invScaleShape = {1};
  void* input1DeviceAddr = nullptr;
  void* input2DeviceAddr = nullptr;
  void* foundInfDeviceAddr = nullptr;
  void* invScaleDeviceAddr = nullptr;
  aclTensor* input1 = nullptr;
  aclTensor* input2 = nullptr;
  aclTensor* foundInf = nullptr;
  aclTensor* invScale = nullptr;
  std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> input2HostData = {7, 8, 9};
  std::vector<float> foundInfHostData = {-1.0f};
  std::vector<float> invScaleHostData = {3.1f};
  // Create the input1 aclTensor.
  ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the input2 aclTensor.
  ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the foundInf aclTensor.
  ret = CreateAclTensor(foundInfHostData, foundInfShape, &foundInfDeviceAddr, aclDataType::ACL_FLOAT, &foundInf);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the invScale aclTensor.
  ret = CreateAclTensor(invScaleHostData, invScaleShape, &invScaleDeviceAddr, aclDataType::ACL_FLOAT, &invScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::vector<aclTensor*> tempInput{input1, input2};
  aclTensorList* tensorListInput = aclCreateTensorList(tempInput.data(), tempInput.size());

  // 3. Call the CANN operator library API, and change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnForeachNonFiniteCheckAndUnscale.
  ret = aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(tensorListInput, foundInf, invScale, &workspaceSize,
                                                             &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnForeachNonFiniteCheckAndUnscale.
  ret = aclnnForeachNonFiniteCheckAndUnscale(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachNonFiniteCheckAndUnscale failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values, and copy the results from the device to the host. Modify the code based on the specific API definition.
  auto size = GetShapeSize(selfShape1);
  std::vector<float> self1Data(size, 0);
  ret = aclrtMemcpy(self1Data.data(), self1Data.size() * sizeof(self1Data[0]), input1DeviceAddr,
                    size * sizeof(self1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy self1 result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("self1 result[%ld] is: %f\n", i, self1Data[i]);
  }

  size = GetShapeSize(selfShape2);
  std::vector<float> self2Data(size, 0);
  ret = aclrtMemcpy(self2Data.data(), self2Data.size() * sizeof(self2Data[0]), input2DeviceAddr,
                    size * sizeof(self2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy self2 result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("self2 result[%ld] is: %f\n", i, self2Data[i]);
  }

  size = GetShapeSize(foundInfShape);
  std::vector<float> foundInfData(size, 0);
  ret = aclrtMemcpy(foundInfData.data(), foundInfData.size() * sizeof(foundInfData[0]), foundInfDeviceAddr,
                    size * sizeof(foundInfData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy foundInf result from device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("foundInf result is: %f\n", foundInfData[0]);

  // 6. Release the aclTensor. Modify the code based on the specific API definition.
  aclDestroyTensorList(tensorListInput);
  aclDestroyTensor(foundInf);
  aclDestroyTensor(invScale);

  // 7. Release the device resources. Modify the code based on the specific API definition.
  aclrtFree(input1DeviceAddr);
  aclrtFree(input2DeviceAddr);
  aclrtFree(foundInfDeviceAddr);
  aclrtFree(invScaleDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
