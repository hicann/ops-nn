# aclnnGroupNormSwishGrad

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/group_norm_swish_grad)

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

Description: Performs backpropagation of [aclnnGroupNormSwish](../../group_norm_swish/docs/aclnnGroupNormSwish_en.md).

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnGroupNormSwishGradGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnGroupNormSwishGrad** is called to perform computation.

```c++
aclnnStatus aclnnGroupNormSwishGradGetWorkspaceSize(
    const aclTensor *dy, 
    const aclTensor *mean, 
    const aclTensor *rstd, 
    const aclTensor *x, 
    const aclTensor *gamma, 
    const aclTensor *beta, 
    int64_t          numGroups, 
    char            *dataFormatOptional, 
    double           swishScale, 
    bool             dgammaIsRequire, 
    bool             dbetaIsRequire, 
    const aclTensor *dxOut, 
    const aclTensor *dgammaOut, 
    const aclTensor *dbetaOut, 
    uint64_t        *workspaceSize, 
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnGroupNormSwishGrad(
    void *         workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnGroupNormSwishGradGetWorkspaceSize

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
          <td>dy (aclTensor*) </td>
          <td>Input</td>
          <td>Gradient calculated in reverse mode.</td>
          <td><ul><li>Empty tensors are not supported. </li><li>The dimension ranges from 2D to 8D. The first dimension is N, and the second dimension is C.</li></ul></td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2-8</td>
          <td>√</td>
      </tr>
      <tr>
          <td>mean (aclTensor*) </td>
          <td>Input</td>
          <td>Second output of forward propagation, indicating the mean value of each group after the input is grouped.</td>
          <td><ul><li>Empty tensors are not supported. </li><li>The data type is the same as that of gamma, and the value of N is the same as that of the 0th dimension of dy.</li></ul></td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
      </tr>
      <tr>
          <td>rstd (aclTensor*) </td>
          <td>Input</td>
          <td>Third output of forward propagation, indicating the reciprocal of the standard deviation of each group after the input is grouped.</td>
          <td><ul><li>Empty tensors are not supported. </li><li>The data type is the same as that of gamma, and the value of N is the same as that of the 0th dimension of dy.</li></ul></td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
      </tr>
      <tr>
          <td>x (aclTensor*)</td>
          <td>Input</td>
          <td>Forward input x.</td>
          <td><ul><li>Empty tensors are not supported. </li><li>The data type and shape are the same as those of dy.</li></ul></td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2-8</td>
          <td>√</td>
      </tr>
      <tr>
          <td>gamma (aclTensor*) </td>
          <td>Input</td>
          <td>Scaling coefficient of each channel.</td>
          <td><ul><li>Empty tensors are not supported. </li><li>The data type and dimension are the same as those of dy. The number of elements must be equal to C.</li></ul></td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
      </tr>
      <tr>
          <td>beta (aclTensor*) </td>
          <td>Input</td>
          <td>Offset coefficient of each channel.</td>
          <td><ul><li>Empty tensors are not supported. </li><li>The data type and dimension are the same as those of dy. The number of elements must be equal to C.</li></ul></td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
      </tr>
      <tr>
          <td>numGroups (int64_t) </td>
          <td>Input</td>
          <td>The C dimension of gradOut is divided into groups.</td>
          <td>The value of group must be greater than 0, and C must be exactly divided by group and the ratio cannot exceed 4000.</td>
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
          <td>swishScale (double) </td>
          <td>Input</td>
          <td>Calculation coefficient.</td>
          <td>The recommended value is 1.0.</td>
          <td>DOUBLE</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dgammaIsRequire (bool) </td>
          <td>Input</td>
          <td>Whether to output dgamma.</td>
          <td>The recommended value is TRUE.</td>
          <td>BOOL</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dbetaIsRequire (bool) </td>
          <td>Input</td>
          <td>Whether to output dbeta.</td>
          <td>The recommended value is TRUE.</td>
          <td>BOOL</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dxOut (aclTensor*) </td>
          <td>Output</td>
          <td>`out` in the formula.</td>
          <td>The data type and shape are the same as those of x.</td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2-8</td>
          <td>x</td>
      </tr>
      <tr>
          <td>dgammaOut (aclTensor*) </td>
          <td>Output</td>
          <td>meanOut in the formula.</td>
          <td>The data type and shape are the same as those of gamma.</td>
          <td>FLOAT16, FLOAT, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>x</td>
      </tr>
      <tr>
          <td>dbetaOut (aclTensor*) </td>
          <td>Output</td>
          <td>rstdOut in the formula.</td>
          <td>The data type and shape are the same as those of gamma.</td>
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
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

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
      <td>The input dy, mean, rstd, x, gamma, beta, dxOut, dgammaOut, or dbetaOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of dy is not supported.</td>
    </tr>
    <tr>
      <td>The data types of mean, rstd, x, gamma, and beta are different from that of dy.</td>
    </tr>
    <tr>
      <td>The data type of dxOut is different from that of dy.</td>
    </tr>
  </tbody></table>

## aclnnGroupNormSwishGrad

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
          <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnGroupNormSwishGradGetWorkspaceSize.</td>
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
  - **aclnnGroupNormSwishGrad** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic computation.

- Input shape restrictions:
    1. numGroups must be greater than 0.
    2. C must be exactly divided by group.
    3. The number of elements in dy is equal to $N * C * HxW$.
    4. The number of elements in mean is equal to $N * group$.
    5. The number of elements in rstd is equal to $N * group$.
    6. The number of elements in x is equal to $N * C * HxW$.
    7. The number of elements in gamma is equal to C.
    8. The number of elements in beta is equal to C.
    9. The ratio of C to group cannot exceed 4000.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_group_norm_swish_grad.h"

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
  std::vector<int64_t> dyShape = {2, 3, 4};
  std::vector<int64_t> meanShape = {2, 1};
  std::vector<int64_t> rstdShape = {2, 1};
  std::vector<int64_t> xShape = {2, 3, 4};
  std::vector<int64_t> gammaShape = {3};
  std::vector<int64_t> betaShape = {3};
  std::vector<int64_t> dxOutShape = {2, 3, 4};
  std::vector<int64_t> dgammaOutShape = {3};
  std::vector<int64_t> dbetaOutShape = {3};
  void* dyDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* rstdDeviceAddr = nullptr;
  void* xDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* dxOutDeviceAddr = nullptr;
  void* dgammaOutDeviceAddr = nullptr;
  void* dbetaOutDeviceAddr = nullptr;
  aclTensor* dy = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* rstd = nullptr;
  aclTensor* x = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* dxOut = nullptr;
  aclTensor* dgammaOut = nullptr;
  aclTensor* dbetaOut = nullptr;
  std::vector<float> dyHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                   13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> meanHostData = {2.0, 2};
  std::vector<float> rstdHostData = {2.0, 2};
  std::vector<float> xHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                  13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> gammaHostData = {2.0, 2, 2};
  std::vector<float> betaHostData = {2.0, 2, 2};
  std::vector<float> dxOutHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                   13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> dgammaOutHostData = {2.0, 2, 2};
  std::vector<float> dbetaOutHostData = {2.0, 2, 2};
  int64_t numGroups = 1;
  char* dataFormatOptional = nullptr;
  float swishScale = 1.0f;
  bool dgammaIsRequire = true;
  bool dbetaIsRequire = true;
  // Create a dy aclTensor.
  ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a mean aclTensor.
  ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a rstd aclTensor.
  ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gamma aclTensor.
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a beta aclTensor.
  ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a dxOut aclTensor.
  ret = CreateAclTensor(dxOutHostData, dxOutShape, &dxOutDeviceAddr, aclDataType::ACL_FLOAT, &dxOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a dgammaOut aclTensor.
  ret = CreateAclTensor(dgammaOutHostData, dgammaOutShape, &dgammaOutDeviceAddr, aclDataType::ACL_FLOAT, &dgammaOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a dbetaOut aclTensor.
  ret = CreateAclTensor(dbetaOutHostData, dbetaOutShape, &dbetaOutDeviceAddr, aclDataType::ACL_FLOAT, &dbetaOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGroupNormSwishGrad.
  ret = aclnnGroupNormSwishGradGetWorkspaceSize(dy, mean, rstd, x, gamma, beta, numGroups, dataFormatOptional, swishScale, dgammaIsRequire, dbetaIsRequire, dxOut, dgammaOut, dbetaOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSwishGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnGroupNormSwishGrad.
  ret = aclnnGroupNormSwishGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSwishGrad failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(dxOutShape);
  ret = aclrtMemcpy(dxOutHostData.data(), dxOutHostData.size() * sizeof(dxOutHostData[0]), dxOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("dxOutHostData[%ld] is: %f\n", i, dxOutHostData[i]);
  }

  size = GetShapeSize(dgammaOutShape);
  ret = aclrtMemcpy(dgammaOutHostData.data(), dgammaOutHostData.size() * sizeof(dgammaOutHostData[0]), dgammaOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("dgammaOutHostData[%ld] is: %f\n", i, dgammaOutHostData[i]);
  }

  size = GetShapeSize(dbetaOutShape);
  ret = aclrtMemcpy(dbetaOutHostData.data(), dbetaOutHostData.size() * sizeof(dbetaOutHostData[0]), dbetaOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("dbetaOutHostData[%ld] is: %f\n", i, dbetaOutHostData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(dy);
  aclDestroyTensor(mean);
  aclDestroyTensor(rstd);
  aclDestroyTensor(x);
  aclDestroyTensor(gamma);
  aclDestroyTensor(beta);
  aclDestroyTensor(dxOut);
  aclDestroyTensor(dgammaOut);
  aclDestroyTensor(dbetaOut);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(dyDeviceAddr);
  aclrtFree(meanDeviceAddr);
  aclrtFree(rstdDeviceAddr);
  aclrtFree(xDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(dxOutDeviceAddr);
  aclrtFree(dgammaOutDeviceAddr);
  aclrtFree(dbetaOutDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
