# aclnnInplaceQuantScatterV2

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/quant_update_scatter)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                         |    √  |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>    |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×   |

## Function

Fuses the [Quantize](https://gitcode.com/cann/ops-nn/blob/master/quant/quantize/docs/aclnnQuantize.md) and [Scatter](https://gitcode.com/cann/ops-nn/blob/master/index/scatter/docs/aclnnInplaceScatterUpdate.md) operators. Quantizes updates along the quantAxis axis: quantScales scales updates, and quantZeroPoints offsets updates. Then, the values in the quantized updates are updated one by one at the corresponding positions in selfRef based on the index tensor indices along the specified axis. Compared with aclnnInplaceQuantScatter, this function has an additional input roundMode.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnInplaceQuantScatterV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnInplaceQuantScatterV2` is called to perform computation.

```cpp
aclnnStatus aclnnInplaceQuantScatterV2GetWorkspaceSize(
  aclTensor       *selfRef,
  const aclTensor *indices,
  const aclTensor *updates,
  const aclTensor *quantScales,
  const aclTensor *quantZeroPoints,
  int64_t          axis,
  int64_t          quantAxis,
  int64_t          reduction,
  const char       *roundMode,
  uint64_t         *workspaceSize,
  aclOpExecutor   **executor);
```

```cpp
aclnnStatus aclnnInplaceQuantScatterV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream);
```

## aclnnInplaceQuantScatterV2GetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
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
      <td>selfRef (aclTensor*)</td>
      <td>Input | Output</td>
      <td>Source data tensor.</td>
      <td>Empty tensors are supported.</td>
      <td>INT8, FLOAT8_E4M3FN, FLOAT_E5M2, HIFLOAT8</td>
      <td>ND</td>
      <td>3-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices (aclTensor*)</td>
      <td>Input</td>
      <td>Index tensor.</td>
      <td><ul><li>When the shape of indices is 1-dimensional, the value range of indices is [0, selfRef.shape(axis) - updates.shape(axis))</li><li>. When the shape of indices is 2-dimensional, the value range of the 0th data of each item in indices is [0, selfRef.shape(0)), and the value range of the 1st data of each item in indices is [0, selfRef.shape(axis) - updates.shape(axis))</li></ul></td>.
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1,2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>updates (aclTensor*)</td>
      <td>Input</td>
      <td>Tensor to be updated.</td>
      <td><ul><li>The number of dimensions of updates must be the same as that of selfRef. The size of the first dimension of updates must be equal to that of indices and cannot be greater than that of selfRef.</li><li>The size of the axis of updates cannot be greater than that of selfRef.</li><li>The size of other dimensions must be the same as that of the corresponding dimensions of selfRef.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantScales (aclTensor*)</td>
      <td>Input</td>
      <td>Quantization scale tensor.</td>
      <td>The number of elements in quantScales must be equal to the size of updates along the quantAxis.</td>
      <td>BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantZeroPoints (aclTensor*)</td>
      <td>Input</td>
      <td>Quantization offset tensor.</td>
      <td>The number of elements in quantZeroPoints must be equal to the size of updates along the quantAxis.</td>
      <td>BFLOAT16, INT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis (int64_t)</td>
      <td>Input</td>
      <td>updates axis used for update.</td>
      <td>Value range: [-len(updates.shape) + 1, -1) or [1, len(updates.shape) - 1).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantAxis (int64_t)</td>
      <td>Input</td>
      <td>Axis used for quantization on updates.</td>
      <td>The value can be -1 or len (updates.shape) - 1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction (int64_t)</td>
      <td>Input</td>
      <td>Data operation mode.</td>
      <td> Currently, the value can only be 1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundMode (char*)</td>
      <td>Input</td>
      Round in the <td>Quantize formula, which specifies the data conversion mode.</td>
      <td><ul><li> supports the {"rint", "round", "hybrid"} mode. If the data type of </li><li>selfRef is INT8/FLOAT8_E4M3FN/FLOAT8_E5M2, only {"rint"} is supported. </li><li>The data type is HIFLOAT8, and {"round", "hybrid"}</li></ul></td> is supported.
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

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 319px">
  <col style="width: 108px">
  <col style="width: 621px">
  </colgroup>
  <thead>
  <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
  </tr></thead>
  <tbody>
  <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The required input, output, or attribute is passed as a null pointer.</td>
  </tr>
  <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types of selfRef, indices, updates, quantScales, and quantZeroPoints are not supported.</td>
  </tr>
  <tr>
    <td>The combinations of selfRef, indices, updates, quantScales, and quantZeroPoints are not supported. For details about the combinations, see the restrictions.</td>
  </tr>
  <tr>
    <td>The number of dimensions of selfRef is inconsistent with that of updates.</td>
  </tr>
  <tr>
    <td>axis, quantAxis, and roundMode are not supported.</td>
  </tr>
  <tr>
    <td>The product model is not supported.</td>
  </tr>
  </tbody>
  </table>

## aclnnInplaceQuantScatterV2

- **Parameters**

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
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnInplaceQuantScatterV2GetWorkspaceSize.</td>
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

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - The default deterministic implementation of aclnnInplaceQuantScatterV2 is used.

- **indices** can only be one- or two-dimensional. If it is two-dimensional, the size of the second dimension must be **2**. Index out-of-bounds is not supported or verified. The **selfRef** data segments mapped by **indices** cannot overlap. If they overlap, the execution results may be different due to multi-core concurrency.
- The input combinations of the data types of **selfRef**, **indices**, **updates**, **quantScales**, and **quantZeroPoints** are as follows:
  - Ascend 950PR/Ascend 950DT:

    |selfRef|indices|updates|quantScales|quantZeroPoints|
    |---|---|---|---|---|
    |INT8, FLOAT8_E4M3FN, FLOAT_E5M2, HIFLOAT8|INT32|BFLOAT16|BFLOAT16|BFLOAT16|
    |INT8, FLOAT8_E4M3FN, FLOAT_E5M2, HIFLOAT8|INT64|BFLOAT16|BFLOAT16|BFLOAT16|
    |INT8, FLOAT8_E4M3FN, FLOAT_E5M2, HIFLOAT8|INT32|FLOAT16|FLOAT32|INT32|
    |INT8, FLOAT8_E4M3FN, FLOAT_E5M2, HIFLOAT8|INT64|FLOAT16|FLOAT32|INT32|

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_scatter_v2.h"

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
  const char* roundMode = "rint";
  // Call the first part of the aclnnInplaceQuantScatterV2 API.
  ret = aclnnInplaceQuantScatterV2GetWorkspaceSize(selfRef, indices, updates, quantScales, quantZeroPoints, axis,
                                                 quantAxis, reduction, roundMode, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnInplaceQuantScatterV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second segment of the aclnnInplaceQuantScatterV2 API.
  ret = aclnnInplaceQuantScatterV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceQuantScatterV2 failed. ERROR: %d\n", ret); return ret);

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
