# aclnnGroupedDynamicBlockQuant

[📄 View Source Code](https://gitcode.com/cann/ops-nn/tree/master/quant/grouped_dynamic_block_quant)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- This API is used to quantize each group at the granularity of basic blocks to FP8/HiFP8 based on the input start value of the group index (groupList), and output the quantization parameter scale (FP32).

- Formula:

  $$
   input\_max = block\_reduce\_max(abs(input))
  $$

  $$
   scale = min(input\_max/FP8\_MAX(HiF8\_MAX), 1/min\_scale)
  $$

  $$
   y = cast\_to\_[HiF8/FP8](input/scale)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. You must call aclnnGroupedDynamicBlockQuantGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call aclnnGroupedDynamicBlockQuant to perform the computation.

```cpp
aclnnStatus aclnnGroupedDynamicBlockQuantGetWorkspaceSize(
  const aclTensor *x, 
  const aclTensor *groupList, 
  double           minScale, 
  char            *roundModeOptional, 
  int64_t          dstType, 
  int64_t          rowBlockSize, 
  int64_t          colBlockSize, 
  int64_t          groupListType, 
  const aclTensor *yOut, 
  const aclTensor *scaleOut, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnGroupedDynamicBlockQuant(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnGroupedDynamicBlockQuantGetWorkspaceSize

- **Parameters**
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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor, corresponding to input in the formula.</td>
      <td>Empty tensors are supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2–3, such as [M, N] and [B, M, N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupList (aclTensor*)</td>
      <td>Input</td>
      <td>Offset of each group on the M axis (cumsum mode).</td>
      <td>Start index of the quantization group. The value must be greater than or equal to 0, and the values must be in non-decreasing order. The last value must be equal to the size of axis -2 of x.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>minScale (double)</td>
      <td>Input</td>
      <td>Minimum scale value for scaleOut computation. It corresponds to min_scale in the formula.</td>
      <td>The value must be greater than or equal to 0.</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*)</td>
      <td>Input</td>
      <td>Indicates the approximation mode in which the high-order bits are cast to the destination data type.</td>
      <td><ul><li>When dstType is set to 35 or 36 and the output yOut data type is FLOAT8_E5M2/FLOAT8_E4M3FN, only {"rint"} is supported.</li><li>When dstType is set to 34 and the output yOut data type is HIFLOAT8, {"round" and "hybrid"} are supported.</li><li>If a null pointer is passed, the "rint" mode is used.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Indicates the data type of yOut after conversion.</td>
      <td>The input value can be {34, 35, 36}, corresponding to the output y data type of {34:HIFLOAT8, 35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rowBlockSize (int64_t)</td>
      <td>Input</td>
      <td>Indicates the quantization granularity on the specified M axis.</td>
      <td>The value can be 1/128/256/512.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>colBlockSize (int64_t)</td>
      <td>Input</td>
      <td>Quantization granularity on the specified axis N.</td>
      <td>The value can be 64, 128, 192, or 256.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType (int64_t)</td>
      <td>Input</td>
      <td>Function type of group_list.</td>
      <td>The value can be 0, corresponding to the cumsum mode.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Quantized output tensor, It corresponds to y in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape dimension is the same as that of x.</li></ul></td>
      <td>HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Quantization scale of each group, corresponding to scale in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>If the shape of input x is [M, N] and the shape of groupList is [g], the shape of the output scaleOut is [(M//rowBlockSize+g), (N/colBlockSize)]. </li><li>If the shape of input x is [B, M, N] and the shape of groupList is [g], the shape of the output scaleOut is [B, (M//rowBlockSize+g), (N/colBlockSize)].</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-3</td>
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
  </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>The input x, groupList, yOut, or scaleOut parameter is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The input or output data format or data type is not supported.</td>
    </tr>
    <tr>
      <td>The input or output data shape is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedDynamicBlockQuant

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
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnGroupedDynamicBlockQuantGetWorkspaceSize API.</td>
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

 - Deterministic description: aclnnGroupedDynamicBlockQuant is implemented in deterministic mode by default.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_grouped_dynamic_block_quant.h"

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
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
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

    // Create an x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a groupList aclTensor.
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT32, &groupList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a y aclTensor.
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a scale aclTensor.
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    const char* roundMode = "rint";
    float minScale = 0.0;
    int64_t rowBlockSize = 1;
    int64_t colBlockSize = 128;
    int64_t groupListType = 0;

    // Call the first part of the aclnnGroupedDynamicBlockQuant API.
    ret = aclnnGroupedDynamicBlockQuantGetWorkspaceSize(x, groupList, minScale, (char *)roundMode, aclDataType::ACL_FLOAT8_E5M2, rowBlockSize, colBlockSize, groupListType, y, scale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicBlockQuantGetWorkspaceSize failed. ERROR: %d\n", ret); 
              return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); 
                return ret);
    }

    // Call the second phase of the aclnnGroupedDynamicBlockQuant API.
    ret = aclnnGroupedDynamicBlockQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicBlockQuant failed. ERROR: %d\n", ret); 
              return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); 
              return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    LOG_PRINT("yOut is: \n");
    PrintOutResult(yShape, &yDeviceAddr);
    LOG_PRINT("scaleOut is: \n");
    PrintOutResult(scaleShape, &scaleDeviceAddr);

    // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(groupList);
    aclDestroyTensor(y);
    aclDestroyTensor(scale);

    // 7. Free device resources.
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
