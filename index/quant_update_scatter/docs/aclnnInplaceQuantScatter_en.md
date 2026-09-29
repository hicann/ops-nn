# aclnnInplaceQuantScatter

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT|√|
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    ×     |

## Function

Quantizes updates on the quantAxis axis, scales updates using quantScales, and offsets updates using quantZeroPoints. Then, the values in the quantized updates are updated one by one at the corresponding positions in selfRef based on the index tensor indices along the specified axis.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnInplaceQuantScatterGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnInplaceQuantScatter** is called to perform computation.

```c++
aclnnStatus aclnnInplaceQuantScatterGetWorkspaceSize(
  aclTensor       *selfRef,
  const aclTensor *indices,
  const aclTensor *updates,
  const aclTensor *quantScales,
  const aclTensor *quantZeroPoints,
  int64_t          axis,
  int64_t          quantAxis,
  int64_t          reduction,
  uint64_t         *workspaceSize,
  aclOpExecutor   **executor)
```

```c++
aclnnStatus aclnnInplaceQuantScatter(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnInplaceQuantScatterGetWorkspaceSize

- **Parameter description**:

  <table style="undefined;table-layout: fixed; width: 1788px"><colgroup>
  <col style="width: 245px">
  <col style="width: 133px">
  <col style="width: 311px">
  <col style="width: 311px">
  <col style="width: 208px">
  <col style="width: 208px">
  <col style="width: 208px">
  <col style="width: 164px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>selfRef</td>
      <td>Input | Output</td>
      <td>Source data tensor.</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Index tensor.</td>
      <td>indices must be in the range of [0, selfRef.shape(axis) - updates.shape(axis)).</td>
      <td>INT32 or INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Data update tensor.</td>
      <td>The number of dimensions of updates must be the same as that of selfRef. The size of the first dimension of updates is equal to that of the first dimension of indices and is not greater than that of the first dimension of selfRef. The size of the axis of updates is not greater than that of the axis of selfRef. The sizes of other dimensions of updates must be the same as those of the corresponding dimensions of selfRef.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>Same as that of selfRef.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantScales</td>
      <td>Input</td>
      <td>Quantization scale tensor.</td>
      <td>The number of elements must be equal to the size of updates along the quantAxis axis.</td>
      <td>BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantZeroPoints</td>
      <td>Optional input</td>
      <td>Quantization offset tensor.</td>
      <td>The number of elements must be equal to the size of updates on the quantAxis axis.</td>
      <td>BFLOAT16, INT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Input</td>
      <td>Axis to be updated in updates.</td>
      <td>Only -2 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantAxis</td>
      <td>Input</td>
      <td>Axis to be quantized in updates.</td>
      <td>The value can be -1 or len(updates.shape) - 1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Input</td>
      <td>Data operation mode.</td>
      <td>The value can be 1 (update).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
  </tbody></table>

  - <term>Atlas inference products</term>
    - The data type BFLOAT16 is not supported.
  - For <term>Atlas inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>:
    - The size of the last dimension of selfRef and updates must be 32-byte aligned.

- **Returns**

    `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter validation. The following error codes may be returned.

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 330px">
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
        <td> ACLNN_ERR_PARAM_NULLPTR </td>
        <td> 161001 </td>
        <td>The required input, output, or attribute is passed as a null pointer.</td>
        </tr>
        <tr>
        <td rowspan="2"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="2"> 161002 </td>
        <td>The combinations of selfRef, indices, updates, quantScales, and quantZeroPoints are not supported. For details about the combinations, see the restrictions.</td>
        </tr>
        <tr>
        <td>The dimensions of selfRef and updates are inconsistent.</td>
        </tr>
    </tbody></table>

## aclnnInplaceQuantScatter

- **Parameters**

    <table>
            <thead>
                <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
            </thead>
            <tbody>
                <tr><td>workspace</td><td>Input</td><td>Address of the workspace to be allocated on the device.</td></tr>
                <tr><td>workspaceSize</td><td>Input</td><td>Size of the workspace allocated on the device, which is obtained by the first API aclnnInplaceQuantScatterGetWorkspaceSize.</td></tr>
                <tr><td>executor</td><td>Input</td><td>The operator executor, which contains the computation process of the operator. </td></tr>
                <tr><td>stream</td><td>Input</td><td>Stream for executing a task. </td></tr>
            </tbody>
    </table>

- **Returns**
  
  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - **aclnnInplaceQuantScatter** defaults to a deterministic implementation.

- The indices can only be 1-dimensional. Index out-of-bounds is not supported and is not verified. The selfRef data segments mapped to indices cannot overlap. If they overlap, the execution results will be different due to multi-core concurrency.
- The input combinations of the data types of **selfRef**, **indices**, **updates**, **quantScales**, and **quantZeroPoints** are as follows:
  - <term>Atlas inference products</term>:

    |selfRef|indices|updates|quantScales|quantZeroPoints|
    |---|---|---|---|---|
    |INT8|INT32|FLOAT16|FLOAT32|INT32|
    |INT8|INT64|FLOAT16|FLOAT32|INT32|

  - <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>, and Ascend 950 PR/Ascend 950 DT:

    |selfRef|indices|updates|quantScales|quantZeroPoints|
    |---|---|---|---|---|
    |INT8|INT32|BFLOAT16|BFLOAT16|BFLOAT16|
    |INT8|INT64|BFLOAT16|BFLOAT16|BFLOAT16|
    |INT8|INT32|FLOAT16|FLOAT32|INT32|
    |INT8|INT64|FLOAT16|FLOAT32|INT32|

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_scatter.h"

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
  // (Boilerplate) Initialize resources.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfRefShape = {1, 1, 32};
  std::vector<int64_t> indicesShape = {1};
  std::vector<int64_t> updatesShape = {1, 1, 32};
  std::vector<int64_t> quantScalesShape = {1, 1, 32};
  std::vector<int64_t> quantZeroPointsShape = {1, 1, 32};
  void* selfRefDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* updatesDeviceAddr = nullptr;
  void* quantScalesDeviceAddr = nullptr;
  void* quantZeroPointsDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* updates = nullptr;
  aclTensor* quantScales = nullptr;
  aclTensor* quantZeroPoints = nullptr;
  std::vector<int8_t> selfRefHostData{32, 0};
  std::vector<int32_t> indicesHostData{0};
  std::vector<float> updatesHostData{32, 1.0};
  std::vector<float> quantScalesHostData{32, 0.5};
  std::vector<float> quantZeroPointsHostData{32, 0.5};
  int64_t axis = -2;
  int64_t quantAxis = -1;
  int64_t reduction = 1;

  // Create a selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_INT8, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an updates aclTensor.
  ret = CreateAclTensor(updatesHostData, updatesShape, &updatesDeviceAddr, aclDataType::ACL_FLOAT16, &updates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a quantScales aclTensor.
  ret = CreateAclTensor(quantScalesHostData, quantScalesShape, &quantScalesDeviceAddr, aclDataType::ACL_FLOAT,
                        &quantScales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a quantZeroPoints aclTensor.
  ret = CreateAclTensor(quantZeroPointsHostData, quantZeroPointsShape, &quantZeroPointsDeviceAddr,
                        aclDataType::ACL_INT32, &quantZeroPoints);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceQuantScatter.
  ret = aclnnInplaceQuantScatterGetWorkspaceSize(selfRef, indices, updates, quantScales, quantZeroPoints, axis,
                                                 quantAxis, reduction, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnInplaceQuantScatterGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceQuantScatter.
  ret = aclnnInplaceQuantScatter(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceQuantScatter failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfRefShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(selfRef);
  aclDestroyTensor(indices);
  aclDestroyTensor(updates);
  aclDestroyTensor(quantScales);
  aclDestroyTensor(quantZeroPoints);

  // 7. Release device resources.
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(updatesDeviceAddr);
  aclrtFree(quantScalesDeviceAddr);
  aclrtFree(quantZeroPointsDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
