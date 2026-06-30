# aclnnDynamicQuantV2

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/quant/dynamic_quant_v2)

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    √     |
| <term>Atlas training series products</term>                             |    √     |

## Function

- Description: Performs per-token symmetric/asymmetric dynamic quantization on the input tensor. In the MOE scenario, **smoothScalesOptional** for each expert is different and is distinguished by the input **groupIndexOptional**.

- Formula:
  - Symmetric quantization:
    - When **smoothScalesOptional** is not provided:

      $$
        scaleOut=row\_max(abs(x))/127
      $$

      $$
        yOut=round(x/scaleOut)
      $$

    - When **smoothScalesOptional** is provided:

      $$
        input = x\cdot smoothScalesOptional
      $$

      $$
        scaleOut=row\_max(abs(input))/127
      $$

      $$
        yOut=round(input/scaleOut)
      $$

  - Asymmetric quantization:
    - When **smoothScalesOptional** is not provided:

      $$
        scaleOut=(row\_max(x) - row\_min(x))/scale\_opt
      $$

      $$
        offset=offset\_opt-row\_max(x)/scaleOut
      $$

      $$
        yOut=round(x/scaleOut+offset)
      $$

    - When **smoothScalesOptional** is provided:

      $$
        input = x\cdot smoothScalesOptional
      $$

      $$
        scaleOut=(row\_max(input) - row\_min(input))/scale\_opt
      $$

      $$
        offset=offset\_opt-row\_max(input)/scaleOut
      $$

      $$
        yOut=round(input/scaleOut+offset)
      $$

  Where **row\_max** denotes computing the maximum value for each row, and **row_min** denotes computing the minimum value for each row. When the type of the output **yOut** is INT8, **scale_opt** is **255.0** and **offset_opt** is **127.0**; when the type of **yOut** is INT4, **scale_opt** is **15.0** and **offset_opt** is **7.0**.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnDynamicQuantV2GetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnDynamicQuantV2** is called to perform computation.

```cpp
aclnnStatus aclnnDynamicQuantV2GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *smoothScalesOptional,
  const aclTensor *groupIndexOptional,
  int64_t          dstType,
  const aclTensor *yOut,
  const aclTensor *scaleOut,
  const aclTensor *offsetOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnDynamicQuantV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDynamicQuantGetWorkspaceSize

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
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input tensor, corresponding to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>The shape must have at least two dimensions.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScalesOptional</td>
      <td>Input</td>
      <td>Smoothing scales for the input, corresponding to `smoothScalesOptional` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of x. </li><li>When groupIndexOptional is not provided, the shape matches the last dimension of x. When groupIndexOptional is provided, the shape is 2D: the first dimension is the number of experts (≤ 1024), and the second dimension matches the last dimension of x.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional</td>
      <td>Input</td>
      <td>Group index for the input.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape has only one dimension.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dstType</td>
      <td>Input</td>
      <td>Enumeration value corresponding to the type of the output y.</td>
      <td><ul><li>If the type of y is INT8, the value is 2; if the type of y is INT4, the value is 29.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>Output</td>
      <td>Quantized output tensor, corresponding to `yOut` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as that of x.</li></ul></td>
      <td>INT4, INT8</td>
      <td>ND</td>
      <td>The shape must have at least two dimensions.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOut</td>
      <td>Output</td>
      <td>Quantization scale, corresponding to `scaleOut` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as the shape of x without its last dimension.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>The shape must have at least one dimension.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetOut</td>
      <td>Output</td>
      <td>Quantization offset, corresponding to `offset` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as the shape of x without its last dimension.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>The shape must have at least one dimension.</td>
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
      <td>Operator executor, containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas inference series products</term> and <term>Atlas training series products</term>:
    - Only FLOAT16 is supported for the input parameter `x`.
    - The input parameters `smoothScalesOptional` and `groupIndexOptional` must be **nullptr**.
    - The input parameter `dstType` must be set to **2**.
    - Only INT8 is supported for the output parameter `yOut`.
  
- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed x or yOut is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type, data format, or dimension of the parameter is not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td>561001</td>
      <td>Failed to create aclOpExecutor internally.</td>
    </tr>
  </tbody></table>

## aclnnDynamicQuantV2

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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnDynamicQuantV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- When the data type of `yOut` is INT4, the last dimensions of `x` and `yOut` must be divisible by 2.
- When the data type of `yOut` is INT32, the last dimension of `x` must be divisible by 8.
- <term>Atlas inference series products</term>: Only 32-bit aligned data is supported on the tail axis. Currently only symmetric quantization is supported, and the BFLOAT16 data type is not supported.
- Deterministic computation:
  - **aclnnDynamicQuantV2** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dynamic_quant_v2.h"

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
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                           *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
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
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs based on API definitions.
    int rowNum = 4;
    int rowLen = 32;
    int groupNum = 4;
    std::vector<int64_t> xShape = {4, 32};
    std::vector<int64_t> smoothShape = {groupNum, rowLen};
    std::vector<int64_t> groupShape = {groupNum};
    std::vector<int64_t> yShape = {4, 32};
    std::vector<int64_t> scaleShape = {4};
    std::vector<int64_t> offsetShape = {4};

    void* xDeviceAddr = nullptr;
    void* smoothDeviceAddr = nullptr;
    void* groupDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* smooth = nullptr;
    aclTensor* group = nullptr;
    aclTensor* y = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;

    std::vector<aclFloat16> xHostData;
    std::vector<aclFloat16> smoothHostData;
    std::vector<int32_t> groupHostData = {2, rowNum};
    std::vector<int8_t> yHostData;
    std::vector<float> scaleHostData;
    std::vector<float> offsetHostData;
    for (int i = 0; i < rowNum; ++i) {
        for (int j = 0; j < rowLen; ++j) {
            float value1 = i * rowLen + j;
            xHostData.push_back(aclFloatToFloat16(value1));
            yHostData.push_back(0);
        }
        scaleHostData.push_back(0);
        offsetHostData.push_back(0);
    }

    for (int m = 0; m < groupNum; ++m) {
        for (int n = 0; n < rowLen; ++n) {
            float value2 = m * rowLen + n;
            smoothHostData.push_back(aclFloatToFloat16(value2));
        }
    }

    // Create an x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a smooth aclTensor.
    ret = CreateAclTensor(smoothHostData, smoothShape, &smoothDeviceAddr, aclDataType::ACL_FLOAT16, &smooth);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a group aclTensor.
    ret = CreateAclTensor(groupHostData, groupShape, &groupDeviceAddr, aclDataType::ACL_INT32, &group);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a y aclTensor.
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT8, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a scale aclTensor.
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an offset aclTensor.
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Modify the API as required.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnDynamicQuantV2.
    ret = aclnnDynamicQuantV2GetWorkspaceSize(x, smooth, group, 2,  y, scale, offset, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnDynamicQuantV2.
    ret = aclnnDynamicQuantV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicQuantV2 failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize the stream and wait for task completion.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    PrintOutResult(yShape, &yDeviceAddr);

    // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(smooth);
    aclDestroyTensor(y);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);

    // 7. Release device resources.
    aclrtFree(xDeviceAddr);
    aclrtFree(smoothDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
