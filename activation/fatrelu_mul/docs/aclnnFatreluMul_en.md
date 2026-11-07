# aclnnFatreluMul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T09:35:22.347Z -->

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

- Description:

  Splits the input tensor into left and right tensors along the last dimension: x1 and x2, performs threshold computation on the left x1, and multiplies the result with x2.

- Formula:

  Given an input tensor x whose last dimension has a length of 2d, the following computation is performed:

  1. Split x into two parts:

     $$
     x_1 = x[..., :d], \quad x_2 = x[..., d:]
     $$
  2. Apply the Threshold activation function to x1, defined as follows:

     $$
     \text{Threshold}(x, \text{threshold}) =
        \begin{cases}
        0 & \text{if } x \leq \text{threshold} \\
        x & \text{if } x > \text{threshold}
        \end{cases}
     $$

     Therefore, the computation is:

     $$
     x_1 = \text{Threshold}(x_1, \text{threshold})
     $$
  3. The final output is the element-wise product of x1 and x2:

     $$
     \text{out} = x_1 \times x_2
     $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnFatreluMulGetWorkspaceSize" to obtain the input parameters and calculate the required workspace size based on the computation process. Then call "aclnnFatreluMul" to perform the computation.

```Cpp
aclnnStatus aclnnFatreluMulGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *threshold,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFatreluMul(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnFatreluMulGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 280px">
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
      <td>Input tensor, that is, x in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The last dimension is an even number.</li><li>For Atlas A2 series, Atlas A3 series, and kirin series products: the last dimension is not greater than 1024, and the product of other dimensions is not greater than 200000.</li><li>For Atlas inference series and Atlas training series products: the last dimension is not greater than 1024, and the product of other dimensions is not greater than 200000.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>Not less than 2</td>
      <td>√</td>
    </tr>
     <tr>
      <td>threshold (aclTensor*)</td>
      <td>Input</td>
      <td>Boundary value for the Threshold computation, corresponding to threshold in the formula.</td>
      <td><ul><li>The data type and data format are the same as those of x.</li><li>Requires that the tensor has only one element.</li><li>NaN and Inf are not supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>Unrestricted</td>
      <td>√</td>
    </tr>
      <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor, that is, out in the formula.</td>
      <td><ul><li>The output data type is the same as that of x.</li><li>The value of the last dimension of the output shape is half that of the input shape.</li><li>The values of other dimensions are the same as those of the x shape.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>Not less than 2</td>
      <td>×</td>
    </tr>
      <tr>
      <td>workspaceSize(uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>x, threshold, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of x is not supported.</td>
    </tr>
    <tr>
      <td>The data type of threshold is not supported.</td>
    </tr>
    <tr>
      <td>The data type of out is not supported.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="4">561002</td>
      <td>The rank of x is less than 2.</td>
    </tr>
    <tr>
      <td>The length of the last dimension of x is not an even number.</td>
    </tr>
    <tr>
      <td>The number of elements in threshold is not 1.</td>
    </tr>
    <tr>
      <td>The shape of out does not match that of x (the leading dimensions are inconsistent, or the last dimension is not half of that of x).</td>
    </tr>
  </tbody></table>

## aclnnFatreluMul

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
      <td>Memory address of the workspace requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device, obtained by the first-phase API aclnnFatreluMulGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnFatreluMul defaults to a deterministic implementation.

- For elements in the gate segment (the first half) of input x that are NaN, the output at the corresponding position is 0.

- In typical scenarios, the last axis is a multiple of 16. When the last axis is not 32-byte aligned, it is recommended to use the small operator splicing logic.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fatrelu_mul.h"

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
  // Boilerplate code: resource initialization
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
  // Call aclrtMalloc to apply for device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device memory.
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
  // 1. (boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definitions.
  std::vector<int64_t> inputShape = {2, 4};

  std::vector<float> inputHostData = {0, 1, 2, 3, 4, 5, 6, 7};

  void* inputDeviceAddr = nullptr;

  aclTensor* input = nullptr;
  // Create the threshold aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the threshold aclTensor.
  std::vector<int64_t> threShape = {1};

  std::vector<float> threData = {0.01};

  void* threDeviceAddr = nullptr;

  aclTensor* threshold = nullptr;
  // Create the threshold aclTensor.
  ret = CreateAclTensor(threData, threShape, &threDeviceAddr, aclDataType::ACL_FLOAT, &threshold);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> outShape = {2, 2};
  std::vector<float> outHostData(2 * 2, 1);
  aclTensor* out = nullptr;
  void* outDeviceAddr = nullptr;
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 16 * 1024 * 1024;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnFatreluMul.
  ret = aclnnFatreluMulGetWorkspaceSize(input, threshold, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFatreluMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnFatreluMul.
  ret = aclnnFatreluMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFatreluMul failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Get the output value and copy the result from device memory to host memory. Modify it based on the specific API definition.
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. Destroy the aclTensor. Modify it based on the specific API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(threshold);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it based on the specific API definition.
  aclrtFree(inputDeviceAddr);
  aclrtFree(threDeviceAddr);
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
