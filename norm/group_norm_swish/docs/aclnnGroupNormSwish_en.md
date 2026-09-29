# aclnnGroupNormSwish

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/group_norm_swish)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Computes the group normalization result **out**, mean value **meanOut**, reciprocal **rstdOut** of the standard deviation, and Swish output of the input **x**.
- Formula:
  - **GroupNorm:**
    Assume $E[x] = \bar{x}$ indicates the mean value of $x$, and $Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$ indicates the variance of $x$. Then:

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

    When **activateSwish** is set to **True**, Swish is computed. In this case, **x** in the Swish formula is **out** obtained by using the GroupNorm formula.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnGroupNormSwishGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnGroupNormSwish** is called to perform computation.

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
          <td>Target tensor to be normalized, x in the yOut calculation formula.</td>
          <td><ul><li>Empty tensors are not supported. </li><li>The dimension ranges from 2D to 8D. The first dimension is N, and the second dimension is C. The 0th and 1st dimensions of x must be greater than 0, and the 1st dimension must be exactly divided by group.</li></ul></td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2-8</td>
          <td>√</td>
      </tr>
      <tr>
          <td>gamma (aclTensor*) </td>
          <td>Input</td>
          <td>Gamma parameter in group normalization, γ in the yOut calculation formula.</td>
          <td><ul><li>Empty tensors are not supported. </li><li>The number of elements must be the same as that of the first dimension of the input x. The data type of gamma and beta must be the same as that of x or FLOAT.</li></ul></td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
      </tr>
      <tr>
          <td>beta (aclTensor*) </td>
          <td>Input</td>
          <td>Beta parameter in group normalization, which is β in the yOut calculation formula.</td>
          <td><ul><li>Empty tensors are not supported. </li><li>The number of elements must be the same as that of the first dimension of the input x. The data types of gamma and beta must be the same, and the data type of gamma and beta must be the same as that of x or FLOAT.</li></ul></td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
      </tr>
      <tr>
          <td>numGroups (int64_t) </td>
          <td>Input</td>
          <td>The C dimension of the input gradOut is divided into groups.</td>
          <td>The value of group must be greater than 0.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dataFormatOptional (char*) </td>
          <td>Input</td>
          <td>Data format.</td>
          <td>The recommended value is NCHW.</td>
          <td>CHAR</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>eps (double) </td>
          <td>Input</td>
          <td>eps value in the formulas for calculating yOut and rstdOut, which prevents the offset of dividing by zero.</td>
          <td>The recommended value is 1.0.</td>
          <td>DOUBLE</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>activateSwish (bool) </td>
          <td>Input</td>
          <td>Whether to support Swish calculation.</td>
          <td>If this parameter is set to true, Swish calculation is performed after groupnorm calculation.</td>
          <td>BOOL</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>swishScale (double) </td>
          <td>Input</td>
          <td>Scale value for Swish calculation.</td>
          <td>The recommended value is 1.0.</td>
          <td>DOUBLE</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>yOut (aclTensor*) </td>
          <td>Output</td>
          <td>Group normalization result.</td>
          <td>The data type and shape are the same as those of x.</td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2-8</td>
          <td>x</td>
      </tr>
      <tr>
          <td>meanOut (aclTensor*) </td>
          <td>Mean value after grouping x</td>
          <td>meanOut in the formula.</td>
          <td>The data type is the same as that of gamma. The shape is (N, numGroups), where N indicates the size of the 0th dimension of x, and numGroups is the input for calculation, indicating that the first dimension of the input x is divided into groups.</td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>x</td>
      </tr>
      <tr>
          <td>rstdOut (aclTensor*) </td>
          <td>Output</td>
          <td>Reciprocal of the standard deviation after grouping x.</td>
          <td>The data type is the same as that of gamma. The shape is (N, numGroups), where N indicates the size of the 0th dimension of x, and numGroups is the input for calculation, indicating that the first dimension of the input x is divided into groups.</td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>  
          <td>x</td>
      </tr>
      <tr>
          <td>workspaceSize (uint64_t*)</td>
          <td>Output</td>
          <td>Size of the workspace to be allocated on the device.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>executor (aclOpExecutor**)</td>
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

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
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
      <td>The input x, gamma, beta, yOut, meanOut, or rstdOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>The data types of x, gamma, beta, yOut, meanOut, or rstdOut are not supported.</td>
    </tr>
  </tbody></table>

## aclnnGroupNormSwish

- **Parameters**
  <table>
  <thead>
      <tr>
          <th>Name</th>
          <th>Input/Output</th>
          <th>Description</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td>workspace</td>
          <td>Input</td>
          <td>Memory address of the workspace to be allocated on the device.</td>
      </tr>
      <tr>
          <td>workspaceSize</td>
          <td>Input</td>
          <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnGroupNormSwishGetWorkspaceSize API.</td>
      </tr>
      <tr>
          <td>executor</td>
          <td>Input</td>
          <td>Operator executor, including the operator computation process.</td>
      </tr>
      <tr>
          <td>stream</td>
          <td>Input</td>
          <td>Stream for executing the task.</td>
      </tr>
  </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation
  - **aclnnGroupNormSwish** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
  // (Fixed writing) Initialize resources.
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
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct inputs and outputs based on the API definition.
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
  double eps = 0.00001;
  bool activateSwish = true;
  double scale = 1.0;
  char* dataFormatOptional = "NCHW";
  // Create an x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gamma aclTensor.
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a beta aclTensor.
  ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &yOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a meanOut aclTensor.
  ret = CreateAclTensor(meanOutHostData, meanOutShape, &meanOutDeviceAddr, aclDataType::ACL_FLOAT, &meanOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a rstdOut aclTensor.
  ret = CreateAclTensor(rstdOutHostData, rstdOutShape, &rstdOutDeviceAddr, aclDataType::ACL_FLOAT, &rstdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGroupNormSwish.
  ret = aclnnGroupNormSwishGetWorkspaceSize(x, gamma, beta, numGroups, dataFormatOptional, eps, activateSwish, scale, yOut, meanOut, rstdOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSwishGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnGroupNormSwish.
  ret = aclnnGroupNormSwish(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSwish failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
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

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(gamma);
  aclDestroyTensor(beta);
  aclDestroyTensor(yOut);
  aclDestroyTensor(meanOut);
  aclDestroyTensor(rstdOut);

  // 7. Release device resources. Modify the code based on the API definition.
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
