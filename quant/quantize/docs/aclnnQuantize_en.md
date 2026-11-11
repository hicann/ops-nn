# aclnnQuantize

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:38:59.731Z -->

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

- Description: Quantizes the input tensor x.
- Formula:

  $$
  out=round((x/scales)+zeroPoints)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnQuantizeGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor containing the computation process, and then call the "aclnnQuantize" API to perform the computation.

```Cpp
aclnnStatus aclnnQuantizeGetWorkspaceSize(
  const aclTensor* x,
  const aclTensor* scales,
  const aclTensor* zeroPoints,
  aclDataType      dtype,
  int32_t          axis,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnQuantize(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantizeGetWorkspaceSize

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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Source data tensor to perform quantization. Corresponds to `x` in the formula.</td>
      <td>Empty tensor is supported.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales (aclTensor*)</td>
      <td>Input</td>
      <td>Scaling factor tensor in the quantization process. Corresponds to `scales` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The size must be 1 or equal to the size of the axis in input x.</li><li>If the dtype of `x` is not FLOAT32, the dtype of scales must be consistent with the dtype of `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the zero-point offset tensor in the quantization process, corresponding to `zeroPoints` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Supports passing in a null pointer.</li><li>The size must be 1 or equal to the size of the axis in input x, and equal to the size of scales.</li></ul></td>
      <td>INT32, INT8, UINT8, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dtype (aclDataType)</td>
      <td>Input</td>
      <td>Specifies the output data type.</td>
      <td>Can be set to ACL_INT8, ACL_UINT8, ACL_INT32, ACL_HIFLOAT8, ACL_FLOAT8_E4M3FN, or ACL_FLOAT8_E5M2.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis (int32_t)</td>
      <td>Input</td>
      <td>Represents the element-wise axis to perform quantization. Other axes are broadcast.</td>
      <td><ul><li>When the sizes of the input scales and zeroPoints are all 1, this parameter is not actually used.</li><li>The supported range is negative values less than the dimension count of input x and greater than or equal to the negative of the dimension count of x.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantization output tensor. Corresponds to `out` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>Its shape must be consistent with `x`, and the data type is specified by the input parameter `dtype`.</li></ul></td>
      <td>INT8, UINT8, INT32, HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device side.</td>
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

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>:
    - Data type:
      - The input parameters `x` and `scales` do not support BFLOAT16 or FLOAT32.
      - The input parameter `zeroPoints` does not support FLOAT32. When the data type is BFLOAT16, the data types of `x` and `scales` are both BFLOAT16.
      - The output `out` supports only INT8, UINT8, and INT32.
    - The input parameter `dtype` supports only ACL_INT8, ACL_UINT8, and ACL_INT32.
  <!-- end id7 -->

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - Data type:
      - The input parameter `zeroPoints` does not support FLOAT32. When the data type is BFLOAT16, the data types of `x` and `scales` are both BFLOAT16.
      - The output `out` supports only INT8, UINT8, and INT32.
    - The input parameter `dtype` supports only ACL_INT8, ACL_UINT8, and ACL_INT32.
  <!-- end id8 -->

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
      <th>Error Code.</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x, scales, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type/data format/dimension of x, scales, zeroPoints, or out is out of the supported range.</td>
    </tr>
    <tr>
      <td>The axis specified by input axis exceeds the number of dimensions of input x.</td>
    </tr>
    <tr>
      <td>dtype is not within the valid value range.</td>
    </tr>
    <tr>
      <td>The sizes of input scales and zeroPoints are not equal.</td>
    </tr>
    <tr>
      <td>When the sizes of input scales and zeroPoints are not 1, they are not equal to the size of the axis specified by input axis.</td>
    </tr>
    <tr>
      <td>The data type of output out is inconsistent with the value of input dtype.</td>
    </tr>
  </tbody></table>

## aclnnQuantize

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
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnQuantizeGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnQuantize defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quantize.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for memory on the device.
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

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (boilerplate) device/stream initialization. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. The construction is custom based on the API.
    std::vector<int64_t> xShape = {4, 2};
    std::vector<int64_t> scalesShape = {2};
    std::vector<int64_t> zeroPointsShape = {2};
    std::vector<int64_t> outShape = {4, 2};
    void* xDeviceAddr = nullptr;
    void* scalesDeviceAddr = nullptr;
    void* zeroPointsDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* scales = nullptr;
    aclTensor* zeroPoints = nullptr;
    aclTensor* out = nullptr;

    aclDataType dataType = ACL_INT32;
    int32_t axis = 1;
    std::vector<float> scalesHostData = {1.0, -3.0};
    std::vector<int32_t> zeroPointsData = {2, 10};
    std::vector<float> xHostData = {0.3382, -0.0919, 0.7564, 0.0234, 3.1024, 1.0761, 0.4228, 1.4621};
    std::vector<int32_t> outHostData = {8, 0};

    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the scales aclTensor.
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the zeroPoints aclTensor.
    ret = CreateAclTensor(zeroPointsData, zeroPointsShape, &zeroPointsDeviceAddr, aclDataType::ACL_INT32, &zeroPoints);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, dataType, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, and replace it with the specific API name
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnQuantize
    ret = aclnnQuantizeGetWorkspaceSize(x, scales, zeroPoints, dataType, axis, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantizeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnQuantize
    ret = aclnnQuantize(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantize failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronously wait for the task execution to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value, copy the result on the device side to the host side, and modify it based on the definition of the specific API
    auto outSize = GetShapeSize(outShape);
    std::vector<int32_t> outData(outSize, 0);
    ret = aclrtMemcpy(
        outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr, outSize * sizeof(outData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < outSize; i++) {
        LOG_PRINT("out[%ld] is: %d\n", i, outData[i]);
    }

    // 6. Release aclTensor and aclScalar, and modify it based on the definition of the specific API
    aclDestroyTensor(x);
    aclDestroyTensor(scales);
    aclDestroyTensor(zeroPoints);
    aclDestroyTensor(out);

    // 7. Release device resources, and modify it based on the definition of the specific API
    aclrtFree(xDeviceAddr);
    aclrtFree(scalesDeviceAddr);
    aclrtFree(zeroPointsDeviceAddr);
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
