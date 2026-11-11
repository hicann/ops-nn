# aclnnGroupedDynamicBlockQuantV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:31:13.187Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Quantizes each group at the basic block granularity based on the starting values of the group indices (groupList), into (FP8/HiFP8), and outputs the quantization parameter scale (FP32).
          Compared with the aclnnGroupedDynamicBlockQuant API, this API allows setting the maximum value of the target data type using dstTypeMax when the target data type is HIFLOAT8.

- Formula:

  $$
   input\_max = block\_reduce\_max(abs(input))
  $$

  $$
   scale = min(input\_max/(FP8\_MAX/HiF8\_MAX/DST\_TYPE\_MAX), 1/min\_scale)
  $$

  $$
   y = cast\_to\_[HiF8/FP8](input/scale)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnGroupedDynamicBlockQuantV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnGroupedDynamicBlockQuantV2" API to perform computation.

```cpp
aclnnStatus aclnnGroupedDynamicBlockQuantV2GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *groupList,
  double           minScale,
  char            *roundModeOptional,
  int64_t          dstType,
  int64_t          rowBlockSize,
  int64_t          colBlockSize,
  int64_t          groupListType,
  double           dstTypeMax,
  const aclTensor *yOut,
  const aclTensor *scaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnGroupedDynamicBlockQuantV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGroupedDynamicBlockQuantV2GetWorkspaceSize

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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor of the operator. Corresponds to input in the formula.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-3, of shape [M, N] and [B, M, N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupList (aclTensor*)</td>
      <td>Input</td>
      <td>Offset of each group on the M dimension (cumsum mode).</td>
      <td>Start indices of quantization groups. Each value must be greater than or equal to 0 and non-decreasing, and the last value must be equal to the size of the -2 axis of x.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>minScale (double)</td>
      <td>Input</td>
      <td>Minimum scale value used in the scaleOut calculation. Corresponds to min_scale in the formula.</td>
      <td>This value must be greater than or equal to 0.</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*)</td>
      <td>Input</td>
      <td>Indicates the rounding mode for the final cast of high-bit data to the target data type.</td>
      <td><ul><li>If dstType is 35/36, corresponding to the output yOut data type FLOAT8_E5M2/FLOAT8_E4M3FN, only {"rint"} is supported;</li><li>If dstType is 34, corresponding to the output yOut data type HIFLOAT8, {"round", "hybrid"} are supported;</li><li>If a null pointer is passed in, the "rint" mode is used.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Indicates the data type of yOut after data conversion.</td>
      <td>Input range: {34, 35, 36}, corresponding to the output data type of y: {34:HIFLOAT8, 35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rowBlockSize (int64_t)</td>
      <td>Input</td>
      <td>Specifies the quantization granularity along the M axis.</td>
      <td>Currently supported values: 1/128/256/512.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>colBlockSize (int64_t)</td>
      <td>Input</td>
      <td>Specifies the quantization granularity along the N axis.</td>
      <td>Currently supported values: 64/128/192/256.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType (int64_t)</td>
      <td>Input</td>
      <td>Specifies the function type of group_list.</td>
      <td>Currently supported value: 0, corresponding to the cumsum mode.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax (double)</td>
      <td>Input</td>
      <td>Maximum value of the target data type.</td>
      <td><ul><li>Currently, only HIFLOAT8 is supported as the target data type, and the value range is [0.0, 32768.0]</li><li>0.0 indicates that the original maximum value of the data type is used.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantized output tensor. Corresponds to y in the formula.</td>
      <td>The dimensions of shape are consistent with those of x.</td>
      <td>HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantization scale for each group, which is scale in the corresponding formula.</td>
      <td><ul><li>If the shape of input x is [M, N] and the shape of groupList is [g], the shape of output scaleOut is [(M//rowBlockSize+g), ceil(N/colBlockSize)].</li><li>If the shape of input x is [B, M, N] and the shape of groupList is [g], the shape of output scaleOut is [B, (M//rowBlockSize+g), ceil(N/colBlockSize)].</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The parameter x, groupList, yOut, or scaleOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The input or output data format or data type is out of the supported range.</td>
    </tr>
    <tr>
      <td>The input or output data shape is out of the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedDynamicBlockQuantV2

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
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnGroupedDynamicBlockQuantV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

 - Determinism note: aclnnGroupedDynamicBlockQuantV2 defaults to a deterministic implementation.
 - Supported use case note: Only the use case where both yOut and scaleOut are empty tensors is supported. Other empty tensor use cases are not supported.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_grouped_dynamic_block_quant_v2.h"

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

  void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
    auto size = GetShapeSize(shape);
    std::vector<int8_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }
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
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to request device memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to device memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    // 1. (Boilerplate) Device/stream initialization. For details, see the acl API documentation.
    // Set deviceId based on the actual device
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs as required by the API
    std::vector<int64_t> xShape = {4, 2};
    std::vector<int64_t> groupListShape = {1};
    std::vector<int64_t> yShape = {4, 2};
    std::vector<int64_t> scaleShape = {5, 1};

    void* xDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* groupList = nullptr;
    aclTensor* y = nullptr;
    aclTensor* scale = nullptr;

    std::vector<aclFloat16> xHostData = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int32_t> groupListHostData = {1};
    std::vector<uint8_t> yHostData(8, 0);
    std::vector<float> scaleHostData = {0, 0, 0, 0, 0};

    // Create the x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the groupList aclTensor
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT32, &groupList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the y aclTensor.
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the scale aclTensor.
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    const char* roundMode = "rint";
    double minScale = 0.0;
    int64_t rowBlockSize = 1;
    int64_t colBlockSize = 128;
    int64_t groupListType = 0;
    double dstTypeMax = 0.0;

    // Call the first-phase API of aclnnGroupedDynamicBlockQuantV2.
    ret = aclnnGroupedDynamicBlockQuantV2GetWorkspaceSize(x, groupList, minScale, (char *)roundMode, aclDataType::ACL_FLOAT8_E5M2, rowBlockSize, colBlockSize, groupListType, dstTypeMax, y, scale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicBlockQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                return ret);
    }

    // Call the second-phase API of aclnnGroupedDynamicBlockQuantV2.
    ret = aclnnGroupedDynamicBlockQuantV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicBlockQuantV2 failed. ERROR: %d\n", ret);
              return ret);

    // 4. (Boilerplate) Wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              return ret);

    // 5. Obtain the output value and copy the result from device memory to host memory. Change it based on the interface definition of the specific API.
    LOG_PRINT("yOut is: \n");
    PrintOutResult(yShape, &yDeviceAddr);
    LOG_PRINT("scaleOut is: \n");
    PrintOutResult(scaleShape, &scaleDeviceAddr);

    // 6. Release the aclTensor and aclScalar. Change it based on the interface definition of the specific API.
    aclDestroyTensor(x);
    aclDestroyTensor(groupList);
    aclDestroyTensor(y);
    aclDestroyTensor(scale);

    // 7. Release the device resources.
    aclrtFree(xDeviceAddr);
    aclrtFree(groupListDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }
```
