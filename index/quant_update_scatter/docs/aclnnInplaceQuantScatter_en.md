# aclnnInplaceQuantScatter

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:33:48.038Z -->

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

First, quantize updates along the quantAxis. quantScales performs scaling on updates, and quantZeroPoints performs offset. Then, update the values at corresponding positions in selfRef one by one along the specified axis based on the index tensor indices, using the quantized values from updates.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnInplaceQuantScatterGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the "aclnnInplaceQuantScatter" API to execute the computation.

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

- **Parameters**:

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
      <td>selfRef</td>
      <td>Input|Output</td>
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
      <td>The value range of indices is [0, selfRef.shape(axis) - updates.shape(axis)).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Update data tensor.</td>
      <td>The number of dimensions of updates must be the same as that of selfRef. Its first dimension size must be equal to the first dimension size of indices and not greater than the first dimension size of selfRef; its axis size must not be greater than the axis size of selfRef; and the sizes of the remaining dimensions must be equal to the corresponding dimension sizes of selfRef.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>Same as selfRef</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantScales</td>
      <td>Input</td>
      <td>Quantization scale tensor.</td>
      <td>The number of elements must equal the axis size of updates along quantAxis.</td>
      <td>BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantZeroPoints</td>
      <td>Optional input</td>
      <td>Quantization offset tensor.</td>
      <td>The number of elements must be equal to the axis size of updates along quantAxis.</td>
      <td>BFLOAT16, INT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Input</td>
      <td>Axis on updates used for updating values.</td>
      <td>Only -2 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantAxis</td>
      <td>Input</td>
      <td>Axis on updates used for quantization.</td>
      <td>Supported values: -1 or len(updates.shape) - 1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Input</td>
      <td>Specifies the data operation mode.</td>
      <td>Supported value: 1 (update).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns an op executor that contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>:
    - BFLOAT16 data type is not supported.
  <!-- end id7 -->
  <!-- npu="A3,910b,310p" id8 -->
  - <term>Atlas inference products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - The last dimension size of selfRef and updates must be 32 Byte aligned.

    - Values outside the documented ranges of `axis`, `quantAxis`, or `reduction` produce warning logs without changing existing return codes or execution behavior. The operation always uses update semantics regardless of `reduction`.

  <!-- end id8 -->

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 330px">
    <col style="width: 140px">
    <col style="width: 762px">
    </colgroup>
    <thead>
        <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td> ACLNN_ERR_PARAM_NULLPTR </td>
        <td> 161001 </td>
        <td>Returns 161001 if a null pointer is passed for a mandatory input, output, or attribute.</td>
        </tr>
        <tr>
        <td rowspan="2"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="2"> 161002 </td>
        <td>The data type combination of selfRef, indices, updates, quantScales, and quantZeroPoints is not supported. For details, see Constraints.</td>
        </tr>
        <tr>
        <td>The dimension counts of selfRef and updates are different.</td>
        </tr>
    </tbody></table>

## aclnnInplaceQuantScatter

- **Parameters**

    <table>
            <thead>
                <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
            </thead>
            <tbody>
                <tr><td>workspace</td><td>Input</td><td>Workspace memory address allocated on the Device.</td></tr>
                <tr><td>workspaceSize</td><td>Input</td><td>Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnInplaceQuantScatterGetWorkspaceSize.</td></tr>
                <tr><td>executor</td><td>Input</td><td> Op executor, which contains the operator computation flow. </td></tr>
                <tr><td>stream</td><td>Input</td><td> Stream used to execute the task. </td></tr>
            </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnInplaceQuantScatter defaults to a deterministic implementation.

- The indices must be one-dimensional. Out-of-bounds indexing is not supported and is not validated. The selfRef data segments mapped by indices must not overlap; otherwise, multiple executions may produce different results due to multi-core concurrency.
- The data type input combinations of selfRef, indices, updates, quantScales, and quantZeroPoints include:

  <!-- npu="310p" id9 -->
  - <term>Atlas inference products</term>:

    | selfRef | Indices | Updates | quantScales | quantZeroPoints |
    |---|---|---|---|---|
    |INT8|INT32|FLOAT16|FLOAT32|INT32|
    |INT8|INT64|FLOAT16|FLOAT32|INT32|

  <!-- end id9 -->
  <!-- npu="950,A3,910b" id10 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:

    | selfRef | Indices | Updates | quantScales | quantZeroPoints |
    |---|---|---|---|---|
    |INT8|INT32|BFLOAT16|BFLOAT16|BFLOAT16|
    |INT8|INT64|BFLOAT16|BFLOAT16|BFLOAT16|
    |INT8|INT32|FLOAT16|FLOAT32|INT32|
    |INT8|INT64|FLOAT16|FLOAT32|INT32|

  <!-- end id10 -->

## Example

For reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. The construction must be customized based on the API.
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
  std::vector<int8_t> selfRefHostData(32, 0);
  std::vector<int32_t> indicesHostData{0};
  std::vector<uint16_t> updatesHostData(32, 15360);  // FP16 1.0 = 0x3C00 = 15360
  std::vector<float> quantScalesHostData(32, 0.5);
  std::vector<int32_t> quantZeroPointsHostData(32, 0);
  int64_t axis = -2;
  int64_t quantAxis = -1;
  int64_t reduction = 1;

  // Create the selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_INT8, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the updates aclTensor.
  ret = CreateAclTensor(updatesHostData, updatesShape, &updatesDeviceAddr, aclDataType::ACL_FLOAT16, &updates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the quantScales aclTensor.
  ret = CreateAclTensor(quantScalesHostData, quantScalesShape, &quantScalesDeviceAddr, aclDataType::ACL_FLOAT,
                        &quantScales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the quantZeroPoints aclTensor.
  ret = CreateAclTensor(quantZeroPointsHostData, quantZeroPointsShape, &quantZeroPointsDeviceAddr,
                        aclDataType::ACL_INT32, &quantZeroPoints);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, and replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceQuantScatter.
  ret = aclnnInplaceQuantScatterGetWorkspaceSize(selfRef, indices, updates, quantScales, quantZeroPoints, axis,
                                                 quantAxis, reduction, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnInplaceQuantScatterGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory using the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceQuantScatter.
  ret = aclnnInplaceQuantScatter(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceQuantScatter failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the results from device memory to host memory. Modify as required by the specific API definition.
  auto size = GetShapeSize(selfRefShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify as required by the specific API definition.
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
