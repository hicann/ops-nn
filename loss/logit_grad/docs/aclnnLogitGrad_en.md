# aclnnLogitGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:26:12.901Z -->

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

  Performs backward propagation of [aclnnLogit](../../logit/docs/aclnnLogit.md).

- Formula:

$$
dx_i=
\begin{cases}
NaN, & \text{if } x < \text{0} \text{ or } x > 1 ,eps <0 \\
0, & \text{if } x < \text{eps} \text{ or } x > 1 - \text{eps},eps \geq 0 \\
\frac{dy_i}{x_i \cdot (1 - x_i)}, & \text{if } \text{eps} \leq x_i \leq 1 - \text{eps} \\
\end{cases}
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnLogitGradGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the computation process, and then call the "aclnnLogitGrad" API to perform the computation.

```Cpp
aclnnStatus aclnnLogitGradGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *dy,
  double           eps,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnLogitGrad(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnLogitGradGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 250px">
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
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor, x in the formula.</td>
      <td>Empty tensors are supported.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>dy (aclTensor*)</td>
      <td>Input</td>
      <td>Gradient of the forward output, dy in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type, data format, and shape are consistent with input x.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>eps (double)</td>
      <td>Input</td>
      <td>Epsilon bound of input to prevent division by zero. It is eps in the formula.</td>
      <td>Recommended value: -1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor, the output dx in the formula.</td>
      <td>The data type, data format, and shape of the output are the same as those of input x.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
     <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the computation process of the operator.</td>
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
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x, dy, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type of x, dy, or out is out of the supported range.</td>
    </tr>
  </tbody></table>

## aclnnLogitGrad

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnLogitGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor containing the operator computation flow.</td>
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
  - aclnnLogitGrad defaults to a deterministic implementation.

- Impact of eps on x and the output:
  - When eps is less than 0 and x is outside the range [0, 1], the output is NaN.
  - When eps is greater than or equal to 0, if x is outside the range [eps, 1 - eps], the output is 0.
  - When eps is greater than 1, the output is NaN; when the value is 1, the output is Inf.
  - When eps is Inf, the output is 0.
  - When eps is NaN, the output is NaN.
- Impact of x values on the output:
  - When x is 0 or 1, the output is Inf.
  - When x is Inf or NaN, the output is NaN.
- When dy is Inf or NaN, the output is Inf or NaN.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_logit_grad.h"

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
  // Boilerplate, resource initialization
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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device memory.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the ACL API.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> xShape = {1, 4};

  std::vector<int64_t> dyShape = {1, 4};

  std::vector<float> xHostData = {0.1, 0.2, 0.3, 0.4};

  std::vector<float> dyHostData = {1.0, 2.0, 3.0, 4.0};

  void* xDeviceAddr = nullptr;

  void* dyDeviceAddr = nullptr;

  aclTensor* x = nullptr;

  aclTensor* dy = nullptr;

  // Create the x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the dy aclTensor.
  ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  double eps = -1.0;
  std::vector<int64_t> dxShape = {1, 4};
  std::vector<float> dxHostData(4, 1);
  aclTensor* dx = nullptr;
  void* dxDeviceAddr = nullptr;
  // Create the dx aclTensor.
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 16 * 1024 * 1024;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnLogitGrad.
  ret = aclnnLogitGradGetWorkspaceSize(x, dy, eps, dx, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogitGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnLogitGrad.
  ret = aclnnLogitGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogitGrad failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values, and copy the results on device memory to host memory. Modify based on the actual API definition.
  PrintOutResult(dxShape, &dxDeviceAddr);

  // 6. Release the aclTensor. Modify based on the actual API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(dy);
  aclDestroyTensor(dx);

  // 7. Release device resources. Modify based on the actual API definition.
  aclrtFree(xDeviceAddr);
  aclrtFree(dyDeviceAddr);
  aclrtFree(dxDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
