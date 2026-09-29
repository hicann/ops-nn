# aclnnDynamicQuantV3

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/quant/dynamic_quant)

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

- Function: Dynamically quantizes the input tensor. In the MOE scenario, **smoothScalesOptional** for each expert is different and is distinguished by the input **groupIndexOptional**. Supports symmetric and asymmetric quantization. Supports per-token, per-tensor, and per-channel quantization modes. Compared with aclnnDynamicQuantV2, the per-tensor and per-channel quantization modes are added and specified by the quantMode parameter.

- Formulas:
  - Symmetric quantization:
    - When **smoothScalesOptional** is not provided:
      $$
        scaleOut=\max_{t}(abs(x))/DTYPE_{MAX}
      $$
      $$
        yOut=round(x/scaleOut)
      $$
    - When **smoothScalesOptional** is provided:
      $$
        input = x\cdot smoothScalesOptional
      $$
      $$
        scaleOut=\max_{t}(abs(input))/DTYPE_{MAX}
      $$
      $$
        yOut=round(input/scaleOut)
      $$
  - Asymmetric quantization:
    - When **smoothScalesOptional** is not provided:
      $$
        scaleOut=(\max_{t}(x) - \min_{t}(x))/(DTYPE_{MAX} - DTYPE_{MIN})
      $$
      $$
        offset=DTYPE_{MAX}-\max_{t}(x)/scaleOut
      $$
      $$
        yOut=round(x/scaleOut+offset)
      $$
    - When **smoothScalesOptional** is provided:
      $$
        input = x\cdot smoothScalesOptional
      $$
      $$
        scaleOut=(\max_{t}(input) - \min_{t}(input))/(DTYPE_{MAX} - DTYPE_{MIN})
      $$
      $$
        offset=DTYPE_{MAX}-\max_{t}(input)/scaleOut
      $$
      $$
        yOut=round(input/scaleOut+offset)
      $$
  $\max_{t}$/$\min_{t}$ indicates the mode of calculating the maximum or minimum value. If quantMode is set to pertoken, t is set to row, indicating that the maximum or minimum value is calculated for each token. If quantMode is set to pertensor, t is set to all, indicating that the maximum or minimum value is calculated for the entire tensor. If quantMode is set to perchannel, t is set to col, indicating that the maximum or minimum value is calculated for each channel. $DTYPE_{MAX}$ is the maximum value of the output type, and $DTYPE_{MIN}$ is the minimum value of the output type.

## Prototype

Each operator has <a href="../../../docs/en/context/two_phase_api.md">two-phase API calls</a>. You must call aclnnDynamicQuantV3GetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call aclnnDynamicQuantV3 to perform the computation.

```Cpp
aclnnStatus aclnnDynamicQuantV3GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *smoothScalesOptional,
  const aclTensor *groupIndexOptional,
  int64_t          dstType,
  bool             isSymmetrical,
  const char      *quantMode,
  const aclTensor *yOut,
  const aclTensor *scaleOut,
  const aclTensor *offsetOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDynamicQuantV3(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDynamicQuantV3GetWorkspaceSize

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
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*) </td>
      <td>Input</td>
      <td>Input tensor, corresponding to `x` in the formula.</td>
      <td><ul><li>When the data type of yOut is INT4, the size of the last dimension of x must be exactly divided by 2.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScalesOptional (aclTensor*) </td>
      <td>Input</td>
      <td>Smoothing scales for the input, corresponding to smoothScalesOptional in the formula.</td>
      <td><ul><li>In the pertoken/pertensor scenario, if groupIndexOptional is not specified, the shape dimension is the last dimension of x. </li><li>If groupIndexOptional is specified, the shape is two-dimensional. The size of the first dimension corresponds to the number of experts, and the value cannot exceed 1024. The size of the second dimension is equal to the size of the last dimension of x. </li><li>In the perchannel scenario, the shape size is the size of the penultimate dimension of x. </li><li>The data type must be the same as that of x.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1, 2</td>
      <td>√</td>
    </tr>
       <tr>
      <td>groupIndexOptional (aclTensor*) </td>
      <td>Input</td>
      <td>Group index for the input. It corresponds to groupIndexOptional in the formula description.</td>
      <td><ul><li>The shape supports only one dimension, and the dimension size is equal to the first dimension of smoothScalesOptional. </li><li>If groupIndexOptional is not nullptr, smoothScalesOptional must be not nullptr.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dstType (int64_t) </td>
      <td>Input</td>
      <td>Type of the output yOut after the specified data is converted, corresponding to DType in the formula.</td>
      <td><ul><li>The input value can be {2, 3, 29, 34, 35, 36}, which corresponds to the data type of the output yOut being {2:INT8, 3:INT32, 29:INT4, 34:HIFLOAT8, 35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN}. </li><li>INT32 is actually eight INT4s concatenated.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isSymmetrical (bool) </td>
      <td>Input</td>
      <td>Whether to perform symmetric quantization.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode (char*) </td>
      <td>Input</td>
      <td>Quantization mode.</td>
      <td><ul><li>Currently, the supported modes are "pertoken", "pertensor", and "perchannel". When quantMode is set to pertensor or perchannel, groupIndexOptional must be set to nullptr.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*) </td>
      <td>Output</td>
      <td> Quantized output tensor. The type is specified by dstType and corresponds to yOut in the formula.</td>
      <td><ul><li>If the data type is INT4, the size of the last dimension must be exactly divisible by 2. </li><li>If the data type is INT32, the last dimension of the shape is 1/8 of the last dimension of x. </li><li>For other data types, the shape is the same as that of x.</li><li></li></ul></td>
      <td>INT4, INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, INT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOut (aclTensor*) </td>
      <td>Output</td>
      <td>Quantization scale, corresponding to scaleOut in the formula.</td>
      If <td><ul><li>quantMode is pertoken, the last dimension is removed from the shape x. If </li><li>quantMode is pertensor, the shape is (1,). </li><li>If quantMode is set to perchannel, the shape of x excludes the penultimate dimension.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offsetOut (aclTensor*) </td>
      <td>Output</td>
      <td>Offset used for asymmetric quantization. It corresponds to offsetOut in the formula.</td>
      <td><ul><li>This parameter is supported only when isSymmetrical is set to false. If isSymmetrical is set to true, offsetOut must be set to nullptr.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*) </td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**) </td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input or output parameter is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type, format, or dimension of the parameter is not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td>561001</td>
      <td>Failed to create aclOpExecutor internally.</td>
    </tr>
  </tbody></table>

## aclnnDynamicQuantV3

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
        <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnDynamicQuantV3GetWorkspaceSize API.</td>
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

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - The default deterministic implementation of aclnnDynamicQuantV3 is used.

  When the data type of yOut is INT4, the last dimensions of x and yOut must be divisible by 2.
  When the data type of yOut is INT32, the last dimension of x must be divisible by 8.
  When groupIndexOptional is specified, the number of experts cannot exceed the product of the dimensions of x excluding the last dimension. The value of groupIndexOptional must be a non-decreasing array of non-negative integers, and the last value must be equal to the product of the dimensions of x excluding the last dimension. If this condition is not met, the result is meaningless.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dynamic_quant_v3.h"

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
  int rowNum = 4;
  int rowLen = 2;
  int groupNum = 2;
  std::vector<int64_t> xShape = {4, 2};
  std::vector<int64_t> smoothShape = {groupNum, rowLen};
  std::vector<int64_t> groupShape = {groupNum};
  std::vector<int64_t> yShape = {4, 2};
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

  // 3. Call the CANN operator library API. Modify the API name to the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  const char* quantMode = "pertoken";

  // Call the first part of the aclnnDynamicQuantV3 API.
  ret = aclnnDynamicQuantV3GetWorkspaceSize(x, smooth, group, 2, false, quantMode, y, scale, offset, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicQuantV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second phase of the aclnnDynamicQuantV3 API.
  ret = aclnnDynamicQuantV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicQuantV3 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  PrintOutResult(yShape, &yDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(smooth);
  aclDestroyTensor(y);
  aclDestroyTensor(scale);
  aclDestroyTensor(offset);

  // 7. Free device resources.
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
