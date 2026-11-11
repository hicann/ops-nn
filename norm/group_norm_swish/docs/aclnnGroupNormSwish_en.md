# aclnnGroupNormSwish

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:10:37.593Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supports
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supports
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supports
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Does not support
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Does not support
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Does not support
<!-- end id6 -->

## Function

- Description: Computes the group normalization result out, mean meanOut, reciprocal of standard deviation rstdOut, and the Swish output of the input x.
- Formula:
  - **GroupNorm:**

    Let $E[x] = \bar{x}$ represent the mean of $x$, and $Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$ represent the variance of $x$, then

    $$
    \left\{
    \begin{array} {rcl}
    yOut& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
    meanOut& &= E[x]\\
    rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
    \end{array}
    \right.
    $$

  - **Swish:**

    $$
    yOut = \frac{x}{1+e^{-scale * x}}
    $$

    When activateSwish is True, Swish is calculated. In this case, x in the Swish formula is the out obtained by the GroupNorm formula.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnGroupNormSwishGetWorkspaceSize" API to obtain the workspace size required for the calculation and the executor that contains the operator calculation process, and then call the "aclnnGroupNormSwish" API to perform the calculation.

```c++
aclnnStatus aclnnGroupNormSwishGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *gamma,
    const aclTensor *beta,
    int64_t          numGroups,
    char            *dataFormatOptional,
    double           eps,
    bool             activateSwish,
    double           swishScale,
    const aclTensor *yOut,
    const aclTensor *meanOut,
    const aclTensor *rstdOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnGroupNormSwish(
    void *         workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupNormSwishGetWorkspaceSize

- **Parameters**
    <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
      <col style="width: 220px">
      <col style="width: 120px">
      <col style="width: 187px">
      <col style="width: 387px">
      <col style="width: 187px">
      <col style="width: 187px">
      <col style="width: 187px">
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
       <td>Target tensor for group normalization, corresponding to x in the Formula of yOut.</td>
       <td><ul><li>Does not support empty tensor.</li><li>Supports 2D to 8D dimensions. Dimension 0 is N, and dimension 1 is C. Dimensions 0 and 1 of x must be greater than 0, and dimension 1 must be divisible by group.</li></ul></td>
       <td>FLOAT16, FLOAT, BFLOAT16</td>
       <td>ND</td>
       <td>2-8</td>
       <td>√</td>
      </tr>
      <tr>
       <td>gamma (aclTensor*)</td>
       <td>Input</td>
       <td>gamma parameter in group normalization, corresponding to γ in the Formula of yOut.</td>
       <td><ul><li>Does not support empty tensors.</li><li>The number of elements must be the same as that of dimension 1 of input x. The data types of gamma and beta must be the same and must be the same as that of x or FLOAT.</li></ul></td>
       <td>FLOAT16, FLOAT, BFLOAT16</td>
       <td>ND</td>
       <td>1</td>
       <td>√</td>
      </tr>
      <tr>
       <td>beta (aclTensor*)</td>
       <td>Input</td>
       <td>beta parameter in group normalization, β in the yOut Formula.</td>
       <td><ul><li>Does not support empty tensors.</li><li>The number of elements must be the same as that of dimension 1 of input x. The data types of gamma and beta must be the same and must be the same as that of x or FLOAT.</li></ul></td>
       <td>FLOAT16, FLOAT, BFLOAT16</td>
       <td>ND</td>
       <td>2</td>
       <td>√</td>
      </tr>
      <tr>
       <td>numGroups (int64_t)</td>
       <td>Input</td>
       <td>The C dimension of input x is divided into group groups.</td>
       <td>group must be greater than 0.</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>dataFormatOptional (char*)</td>
       <td>Input</td>
       <td>Data format.</td>
       <td>Recommended value: NCHW.</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>eps (double)</td>
       <td>Input</td>
       <td>Offset that prevents division by zero. eps in the calculation formulas of yOut and rstdOut.</td>
       <td>Recommended value: 1e-5.</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>activateSwish (bool)</td>
       <td>Input</td>
       <td>Whether swish calculation is supported.</td>
       <td>If set to true, the swish calculation is performed after the group normalization calculation.</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>swishScale (double)</td>
       <td>Input</td>
       <td>The scale value used in the Swish calculation.</td>
       <td>Recommended value: 1.0.</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>yOut (aclTensor*)</td>
       <td>Output</td>
       <td>Group normalization result.</td>
       <td>Data type and shape are the same as those of x.</td>
       <td>FLOAT16, FLOAT, BFLOAT16</td>
       <td>ND</td>
       <td>2-8</td>
       <td>x</td>
      </tr>
      <tr>
       <td>meanOut (aclTensor*)</td>
       <td>Output</td>
       <td>meanOut in the formula.</td>
       <td>The data type is the same as gamma. The shape is (N, numGroups), where N indicates the size of dimension 0 of x, and numGroups is a calculation input, indicating that dimension 1 of input x is divided into group groups.</td>
       <td>FLOAT16, FLOAT, BFLOAT16</td>
       <td>ND</td>
       <td>2</td>
       <td>x</td>
      </tr>
      <tr>
       <td>rstdOut (aclTensor*)</td>
       <td>Output</td>
       <td>The reciprocal of the standard deviation after grouping x.</td>
       <td>Same data type as gamma. The shape is (N, numGroups), where N is the size of dimension 0 of x, and numGroups is the calculation input, indicating that dimension 1 of input x is divided into group groups.</td>
       <td>FLOAT16, FLOAT, BFLOAT16</td>
       <td>ND</td>
       <td>2</td>
       <td>x</td>
      </tr>
      <tr>
       <td>workspaceSize (uint64_t*)</td>
       <td>Output</td>
       <td>Returns the workspace size to be applied on the Device side.</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>executor (aclOpExecutor**)</td>
       <td>Output</td>
       <td>Returns the op executor, which contains the operator calculation process.</td>
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

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>Returned when x, gamma, beta, yOut, meanOut, or rstdOut is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types of x, gamma, beta, yOut, meanOut, and rstdOut are not in the supported range.</td>
    </tr>
  </tbody></table>

## aclnnGroupNormSwish

- **Parameters**
  <table>
  <thead>
      <tr>
       <th>Parameter</th>
       <th>Input/Output</th>
       <th>Description</th>
      </tr>
  </thead>
  <tbody>
      <tr>
       <td>workspace</td>
       <td>Input</td>
       <td>Workspace memory address applied on the Device side.</td>
      </tr>
      <tr>
       <td>workspaceSize</td>
       <td>Input</td>
       <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnGroupNormSwishGetWorkspaceSize.</td>
      </tr>
      <tr>
       <td>executor</td>
       <td>Input</td>
       <td>Op executor, which contains the operator calculation flow.</td>
      </tr>
      <tr>
       <td>stream</td>
       <td>Input</td>
       <td>Stream for executing tasks.</td>
      </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation
  - aclnnGroupNormSwish defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_group_norm_swish.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to apply for memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host-side data to device-side memory.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API guide.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. The construction needs to be customized based on the API.
  std::vector<int64_t> xShape = {2, 3, 4};
  std::vector<int64_t> gammaShape = {3};
  std::vector<int64_t> betaShape = {3};
  std::vector<int64_t> outShape = {2, 3, 4};
  std::vector<int64_t> meanOutShape = {2, 1};
  std::vector<int64_t> rstdOutShape = {2, 1};
  void* xDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* meanOutDeviceAddr = nullptr;
  void* rstdOutDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* yOut = nullptr;
  aclTensor* meanOut = nullptr;
  aclTensor* rstdOut = nullptr;
  std::vector<float> xHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                     13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> gammaHostData = {2.0, 2, 2};
  std::vector<float> betaHostData = {2.0, 2, 2};
  std::vector<float> outHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> meanOutHostData = {2.0, 2};
  std::vector<float> rstdOutHostData = {2.0, 2};

  int64_t numGroups = 1;
  double eps = 1e-5;
  bool activateSwish = true;
  double scale = 1.0;
  char* dataFormatOptional = "NCHW";
  // Create the x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the gamma aclTensor.
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the beta aclTensor.
  ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &yOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the meanOut aclTensor.
  ret = CreateAclTensor(meanOutHostData, meanOutShape, &meanOutDeviceAddr, aclDataType::ACL_FLOAT, &meanOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the rstdOut aclTensor.
  ret = CreateAclTensor(rstdOutHostData, rstdOutShape, &rstdOutDeviceAddr, aclDataType::ACL_FLOAT, &rstdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGroupNormSwish.
  ret = aclnnGroupNormSwishGetWorkspaceSize(x, gamma, beta, numGroups, dataFormatOptional, eps, activateSwish, scale, yOut, meanOut, rstdOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSwishGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnGroupNormSwish.
  ret = aclnnGroupNormSwish(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSwish failed. ERROR: %d\n", ret); return ret);
  // 4. (boilerplate) Synchronously wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output values, and copy the results on the device side to the host side. Modify the code according to the definition of the specific API.
  auto size = GetShapeSize(outShape);
  std::vector<float> outResultData(size, 0);
  ret = aclrtMemcpy(outResultData.data(), outResultData.size() * sizeof(outResultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("outResultData[%ld] is: %f\n", i, outResultData[i]);
  }

  size = GetShapeSize(meanOutShape);
  std::vector<float> meanResultData(size, 0);
  ret = aclrtMemcpy(meanResultData.data(), meanResultData.size() * sizeof(meanResultData[0]), meanOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("meanResultData[%ld] is: %f\n", i, meanResultData[i]);
  }

  size = GetShapeSize(rstdOutShape);
  std::vector<float> rstdResultData(size, 0);
  ret = aclrtMemcpy(rstdResultData.data(), rstdResultData.size() * sizeof(rstdResultData[0]), rstdOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("rstdResultData[%ld] is: %f\n", i, rstdResultData[i]);
  }

  // 6. Release aclTensor and aclScalar, which needs to be modified based on the actual API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(gamma);
  aclDestroyTensor(beta);
  aclDestroyTensor(yOut);
  aclDestroyTensor(meanOut);
  aclDestroyTensor(rstdOut);

  // 7. Release device resources, which needs to be modified based on the actual API definition.
  aclrtFree(xDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(meanOutDeviceAddr);
  aclrtFree(rstdOutDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
