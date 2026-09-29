# aclnnGroupNormSiluV2

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/group_norm_silu)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Description: Computes the group normalization result **out**, mean value **meanOut**, reciprocal **rstdOut** of the standard deviation, and SiLU output of the input **self**.
- Formula:
  - **GroupNorm:**
  Assume $E[x] = \bar{x}$ indicates the mean value of $x$, and $Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$ indicates the variance of $x$. Then:
  $$
  \left\{
  \begin{array} {rcl}
  out& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
  meanOut& &= E[x]\\
  rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
  \end{array}
  \right.
  $$
  - **Silu:**
  $$
  out = \frac{x}{1+e^{-x}}
  $$
    If **activateSilu** is set to **True**, Silu is computed. In this case, **x** in the Silu formula is **out** obtained by using the GroupNorm formula.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnGroupNormSiluV2GetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnGroupNormSiluV2** is called to perform computation.

```c++
aclnnStatus aclnnGroupNormSiluV2GetWorkspaceSize(
    const aclTensor* self, 
    const aclTensor* gamma, 
    const aclTensor* beta, 
    int64_t          group, 
    double           eps, 
    bool             activateSilu, 
    aclTensor*       out, 
    aclTensor*       meanOut, 
    aclTensor*       rstdOut, 
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor);
```

```c++
aclnnStatus aclnnGroupNormSiluV2(
    void *         workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnGroupNormSiluV2GetWorkspaceSize

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
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>X in the formula.</td>
        <td>-</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>2 to 8, where the first dimension is N and the second dimension is C</td>.
        <td>√</td>
    </tr>
    <tr>
        <td>gamma (aclTensor*) </td>
        <td>Input</td>
        <td>γ in the formula.</td>
        <td>The data type is the same as that of self or FLOAT. The number of elements must be the same as the size of the second dimension of the input self.</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>beta (aclTensor*) </td>
        <td>Input</td>
        <td>β in the formula.</td>
        <td>The data type is the same as that of self or FLOAT. The number of elements must be the same as the size of the second dimension of the input self.</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>group (int64_t) </td>
        <td>Input</td>
        <td>The second dimension of the input self is divided into groups.</td>
        <td>The number of groups must be exactly divisible by the first dimension of self.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>eps (double) </td>
        <td>Input</td>
        <td>eps in the formula.</td>
        <td>eps must be greater than 0</td>.
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>activateSilu (bool) </td>
        <td>Input</td>
        <td>Whether to enable silu calculation.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>`out` in the formula.</td>
        <td>The data type is the same as that of self.</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>Same as that of self</td>
        <td>x</td>
    </tr>
    <tr>
        <td>meanOut (aclTensor*) </td>
        <td>Output</td>
        <td>meanOut in the formula.</td>
        <td>The data type is the same as that of self. In the shape, N indicates the size of the first dimension of self.</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>(N, group)</td>
        <td>x</td>
    </tr>
    <tr>
        <td>rstdOut (aclTensor*) </td>
        <td>Output</td>
        <td>rstdOut in the formula.</td>
        <td>The data type is the same as that of self. In the shape, N indicates the size of the first dimension of self.</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>(N, group)</td>  
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

  - For the <term>Atlas inference products</term>, the data type BFLOAT16 is not supported.

  - For the Ascend 950PR/Ascend 950DT, the data types of meanOut and rstdOut must be the same as those of gamma and beta.

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter validation. The following error codes may be returned.

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
      <td>The required input, output, or attribute is passed as a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The input and output data types are not supported.</td>
    </tr>
    <tr>
      <td>The input and output parameters do not meet the constraints specified in the parameter description.</td>
    </tr>
  </tbody></table>

## aclnnGroupNormSiluV2

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
        <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnGroupNormSiluV2GetWorkspaceSize.</td>
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

- Deterministic computing: The aclnnGroupNormSiluV2 operator is implemented in deterministic mode by default.

- Input shape restrictions:
    1. The size of the second dimension of self must be exactly divided by group.
    2. The shapes of meanOut and rstdOut must be (N, group), where N is the size of the first dimension of self.
- Input attribute restriction: eps > 0

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_group_norm_silu.h"

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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2.Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {2, 3, 4};
  std::vector<int64_t> gammaShape = {3};
  std::vector<int64_t> betaShape = {3};
  std::vector<int64_t> outShape = {2, 3, 4};
  std::vector<int64_t> meanOutShape = {2, 1};
  std::vector<int64_t> rstdOutShape = {2, 1};
  void* selfDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* meanOutDeviceAddr = nullptr;
  void* rstdOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* out = nullptr;
  aclTensor* meanOut = nullptr;
  aclTensor* rstdOut = nullptr;
  std::vector<float> selfHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                     13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> gammaHostData = {2.0, 2, 2};
  std::vector<float> betaHostData = {2.0, 2, 2};
  std::vector<float> outHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> meanOutHostData = {2.0, 2};
  std::vector<float> rstdOutHostData = {2.0, 2};

  int64_t group = 1;
  double eps = 0.00001;
  bool activateSilu = true;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> selfTensorPtr(self, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gamma aclTensor.
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gammaTensorPtr(gamma, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> gammaDeviceAddrPtr(gammaDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a beta aclTensor.
  ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> betaTensorPtr(beta, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> betaDeviceAddrPtr(betaDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a meanOut aclTensor.
  ret = CreateAclTensor(meanOutHostData, meanOutShape, &meanOutDeviceAddr, aclDataType::ACL_FLOAT, &meanOut);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> meanOutTensorPtr(meanOut, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> meanOutDeviceAddrPtr(meanOutDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a rstdOut aclTensor.
  ret = CreateAclTensor(rstdOutHostData, rstdOutShape, &rstdOutDeviceAddr, aclDataType::ACL_FLOAT, &rstdOut);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> rstdOutTensorPtr(rstdOut, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> rstdOutDeviceAddrPtr(rstdOutDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGroupNormSiluV2.
  ret = aclnnGroupNormSiluV2GetWorkspaceSize(self, gamma, beta, group, eps, activateSilu, out, meanOut, rstdOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSiluV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnGroupNormSiluV2.
  ret = aclnnGroupNormSiluV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSiluV2 failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
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
  aclDestroyTensor(self);
  aclDestroyTensor(gamma);
  aclDestroyTensor(beta);
  aclDestroyTensor(out);
  aclDestroyTensor(meanOut);
  aclDestroyTensor(rstdOut);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(selfDeviceAddr);
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
