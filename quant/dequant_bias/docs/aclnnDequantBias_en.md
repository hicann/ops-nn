# aclnnDequantBias

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:08:55.230Z -->

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Performs dequantization on the input x and converts the input INT32 data into FLOAT16/BFLOAT16 output.
- Formula:

  $$
  y = A \times \text{weight\_scale} \times \text{activate\_scale}
  $$
  $$
    y = (A + \text{bias}) \times \text{weight\_scale} \times \text{activate\_scale}

  $$
  $$
    y = A \times \text{weight\_scale} \times \text{activate\_scale} + \text{bias}

  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnDequantBiasGetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size, and then call the "aclnnDequantBias" API to execute the computation.

```Cpp
aclnnStatus aclnnDequantBiasGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weightScale,
  const aclTensor *activateScaleOptional,
  const aclTensor *biasOptional,
  int64_t          outputDtype,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDequantBias(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnDequantBiasGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 301px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 320px">
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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the input tensor of the dequantization operation, which is A in the formula.</td>
    <td><ul><li>Supports empty tensors (empty input returns empty output).</li><li>The shape is [M, N].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>weightScale (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the weight of the multiplication on the N dimension of the dequantization operation input, which is weight_scale in the formula.</td>
      <td><ul><li>Supports empty tensors (weightScale can be empty when x is empty).</li><li>shape is [N], and the length is the same as the length of dimension N of x.</li></ul></td>
      <td>FLOAT, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
     <tr>
      <td>activateScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the weight of the multiplication on the M dimension of the dequantization operation input, which is the input activate_scale in the formula.</td>
      <td><ul><li>shape is [M], and the length is the same as the length of dimension M of x.</li><li>Optional input: passing an empty Tensor means it does not participate in the computation (equivalent to passing a null pointer); when provided, the shape must be [M].</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the weight added to the N dimension of the dequantization operation input, which is the input bias in the formula.</td>
      <td><ul><li>The shape is [N], and the length is the same as the N dimension length of x.</li><li>Optional input: passing an empty Tensor means it does not participate in the computation (equivalent to passing a null pointer); when provided, the shape must be [N].</li></ul></td>
      <td>FLOAT, BFLOAT16, FLOAT16, INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
      <tr>
      <td>outputDtype (int64_t)</td>
      <td>Input</td>
      <td>Represents the data type of output out.</td>
      <td><ul><li>The value is [1, 27]. The value 1 represents the output type FLOAT16, and the value 27 represents the output type BFLOAT16.</li><li>Set this parameter to 1 when the weightScale data type is FLOAT.</li><li>Set this parameter to 27 when the weightScale data type is BFLOAT16.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the output Tensor of the dequantization operation, that is, output y in the formula.</td>
      <td>The shape is [M, N].</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied for on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor(aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
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

<table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
<col style="width: 319px">
<col style="width: 108px">
<col style="width: 621px">
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
    <td>The tensor of input x, weightScale, or output out is a null pointer.</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data type of input x, weightScale, or output out is not in the supported range.</td>
  </tr>
  <tr>
    <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="2">561002</td>
    <td>The data type of inputs activateScaleOptional and biasOptional is not in the supported range(for example,M is greater than 25000).</td>
  </tr>
  <tr>
    <td>The values of N and M in the shapes of the input and output do not meet the parameter requirements and constraints.</td>
  </tr>
</tbody>
</table>

## aclnnDequantBias

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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnDequantBiasGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
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
  - aclnnDequantBias defaults to a deterministic implementation.

- N and M in the shapes of input and output parameters must be positive integers, and the value of M must be less than or equal to 25000.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dequant_bias.h"

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
  std::vector<int16_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate for resource initialization.
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
  // Call aclrtMemcpy to copy data from the host to the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of a contiguous tensor.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. The construction must be customized based on the API.
  std::vector<int64_t> inputShape = {40, 256};
  std::vector<int64_t> weightShape = {256};
  std::vector<int64_t> activationShape = {40};
  std::vector<int64_t> biasShape = {256};

  std::vector<int32_t> inputHostData(40 * 256, 1);
  std::vector<float> weightHostData(256, 2.0f);
  std::vector<float> activationHostData(40, 2.0f);
  std::vector<float> biasHostData(256, 2.0f);

  void* inputDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* activationDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;

  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* activation = nullptr;
  aclTensor* bias = nullptr;

  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_INT32, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(activationHostData, activationShape, &activationDeviceAddr, aclDataType::ACL_FLOAT, &activation);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);


  std::vector<int64_t> yShape = {40,256};
  std::vector<int16_t> yHostData(40 * 256, 9);
  aclTensor* y = nullptr;
  void* yDeviceAddr = nullptr;


  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnDequantBias.
  ret = aclnnDequantBiasGetWorkspaceSize(input, weight, activation, bias,
      1, y, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantBiasGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnDequantBias.
  ret = aclnnDequantBias(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantBias failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to the host. Modify it based on the specific API definition.
  PrintOutResult(yShape, &yDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify them based on the specific API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(y);

  // 7. Release device resources. Modify it based on the specific API definition.
  aclrtFree(inputDeviceAddr);
  aclrtFree(yDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
