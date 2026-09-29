# aclnnGroupNormSiluQuant

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    x     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |     ×    |
| <term>Atlas training products</term>                             |    ×     |

## Function

- This API is used to calculate the group normalization of the input self, output the mean value meanOut, the reciprocal of the standard deviation rstdOut, and the quantization result out of the silu output.
- Formula:
  - **GroupNorm:**
  Assume $E[x] = \bar{x}$ indicates the mean value of $x$, and $Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$ indicates the variance of $x$. Then:
  
  $$
  \left\{
  \begin{array} {rcl}
  groupNormOut& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
  meanOut& &= E[x]\\
  rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
  \end{array}
  \right.
  $$

  - **Silu:**

  $$
  siluOut = \frac{groupNormOut}{1+e^{-groupNormOut}}
  $$

  - **Quant:**

  $$
  out = round(siluOut / quantScale)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. You must call aclnnGroupNormSiluQuantGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator execution process, and then call aclnnGroupNormSiluQuant to perform the computation.

```c++
aclnnStatus aclnnGroupNormSiluQuantGetWorkspaceSize(
    const aclTensor* self, 
    const aclTensor* gammaOptional, 
    const aclTensor* betaOptional, 
    const aclTensor* quantScale, 
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
aclnnStatus aclnnGroupNormSiluQuant(
    void *         workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnGroupNormSiluQuantGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 187px">
    <col style="width: 121px">
    <col style="width: 287px">
    <col style="width: 387px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 146px">
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
        <td>self</td>
        <td>Input</td>
        <td>X in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2 to 8, where the 0th dimension is N and the 1st dimension is C</td>.
        <td>√</td>
    </tr>
    <tr>
        <td>gammaOptional</td>
        <td>Optional input</td>
        <td>γ in the formula.</td>
        <td>If this parameter is left empty, the default value of the element is 1. The data type is the same as that of self. The number of elements must be the same as that of the first dimension of the input self.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>betaOptional</td>
        <td>Optional input</td>
        <td>β in the formula.</td>
        <td>If this parameter is left empty, the default value of the element is 1. The data type is the same as that of self. The number of elements must be the same as that of the first dimension of the input self.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>quantScale</td>
        <td>Input</td>
        <td>quantScale in the formula.</td>
        <td>The number of elements must be 1 or the same as the first dimension of the input self.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>group</td>
        <td>Input</td>
        <td>The first dimension of the input self is divided into groups.</td>
        <td>The number of groups must be exactly divisible by the first dimension of self.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>eps</td>
        <td>Input</td>
        <td>eps in the formula.</td>
        <td>eps must be greater than 0.</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>activateSilu</td>
        <td>Input</td>
        <td>Whether to enable silu calculation.</td>
        <td>Currently, only enabling is supported.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>out</td>
        <td>Output</td>
        <td>Quantized result, which is out in the formula.</td>
        <td>-</td>
        <td>INT8</td>
        <td>ND</td>
        <td>Same as that of self</td>
        <td>-</td>
    </tr>
    <tr>
        <td>meanOut</td>
        <td>Output</td>
        <td>meanOut in the formula.</td>
        <td>The data type is the same as that of self. The value of N in the shape is the same as that of the 0th dimension of self.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(N, group)</td>
        <td>-</td>
    </tr>
    <tr>
        <td>rstdOut</td>
        <td>Output</td>
        <td>rstdOut in the formula.</td>
        <td>The data type is the same as that of self. The value of N in the shape is the same as that of the 0th dimension of self.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(N, group)</td>  
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

## aclnnGroupNormSiluQuant

- **Parameters**

      <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
        <col style="width: 173px">
        <col style="width: 112px">
        <col style="width: 668px">
        </colgroup>
            <thead>
                <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
            </thead>
            <tbody>
                <tr><td>workspace</td><td>Input</td><td>Address of the workspace to be allocated on the device.</td></tr>
                <tr><td>workspaceSize</td><td>Input</td><td>Size of the workspace allocated on the device, which is obtained by the first API aclnnGroupNormSiluQuantGetWorkspaceSize.</td></tr>
                <tr><td>executor</td><td>Input</td><td>The operator executor, which contains the computation process of the operator. </td></tr>
                <tr><td>stream</td><td>Input</td><td>Stream for executing a task. </td></tr>
            </tbody>
        </table>

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - The aclnnGroupNormSiluQuant is implemented in deterministic mode by default.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_group_norm_silu_quant.h"

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
  // (Fixed writing) Initialize AscendCL.
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
  std::vector<int64_t> selfShape = {1, 2, 4, 4}; // primary input
  std::vector<int64_t> gammaShape = {2}; // gamma parameter
  std::vector<int64_t> betaShape = {2}; // beta parameter 
  std::vector<int64_t> quantScaleShape = {1}; // quantization scaling factor
  std::vector<int64_t> outShape = {1, 2, 4, 4}; // quantization output
  std::vector<int64_t> meanOutShape = {1, 2}; // mean output [N, G]
  std::vector<int64_t> rstdOutShape = {1, 2}; // reciprocal standard deviation output [N, G]

  void* selfDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* quantScaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* meanOutDeviceAddr = nullptr;
  void* rstdOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* quantScale = nullptr;
  aclTensor* out = nullptr;
  aclTensor* meanOut = nullptr;
  aclTensor* rstdOut = nullptr;
  // Use uint16_t to store the bit patterns of BF16/FP16.
  std::vector<uint16_t> selfHostData = {
  FP16 bit patterns of 0x3C00, 0x4000, 0x4200, 0x4400, // 1.0, 2.0, 3.0, 4.0
  0x4500, 0x4600, 0x4700, 0x4800, // FP16 bit patterns of 5.0, 6.0, 7.0, and 8.0
  FP16 bit patterns of 0x3C00, 0x4000, 0x4200, 0x4400, // 1.0, 2.0, 3.0, 4.0
  0x4500, 0x4600, 0x4700, 0x4800, // FP16 bit patterns of 5.0, 6.0, 7.0, and 8.0
  FP16 bit patterns of 0x3C00, 0x4000, 0x4200, 0x4400, // 1.0, 2.0, 3.0, 4.0
  0x4500, 0x4600, 0x4700, 0x4800, // FP16 bit patterns of 5.0, 6.0, 7.0, and 8.0
  FP16 bit patterns of 0x3C00, 0x4000, 0x4200, 0x4400, // 1.0, 2.0, 3.0, 4.0
  0x4500, 0x4600, 0x4700, 0x4800 // FP16 bit patterns of 5.0, 6.0, 7.0, and 8.0
  };
  FP16 bit patterns of std::vector<uint16_t> gammaHostData = {0x3C00, 0x3C00}; // 1.0, 1.0
  FP16 mode of std::vector<uint16_t> betaHostData = {0x0000, 0x0000}; // 0.0, 0.0
  std::vector<float> quantScaleHostData = {1.0};
  std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // Initialized to 0, which is actually filled by the operator.
  // The statistics output is also in half-precision format.
  std::vector<uint16_t> meanOutHostData = {0x0000, 0x0000}; // is initialized to 0.
  std::vector<uint16_t> rstdOutHostData = {0x0000, 0x0000}; // is initialized to 0.

  int64_t group = 2;
  double eps = 0.00001;
  bool activateSilu = true;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gamma aclTensor.
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a beta aclTensor.
  ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a quantScale aclTensor.
  ret = CreateAclTensor(quantScaleHostData, quantScaleShape, &quantScaleDeviceAddr, aclDataType::ACL_FLOAT, &quantScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret); // Create out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a meanOut aclTensor.
  ret = CreateAclTensor(meanOutHostData, meanOutShape, &meanOutDeviceAddr, aclDataType::ACL_FLOAT16, &meanOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a rstdOut aclTensor.
  ret = CreateAclTensor(rstdOutHostData, rstdOutShape, &rstdOutDeviceAddr, aclDataType::ACL_FLOAT16, &rstdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first API of aclnnGroupNormSiluQuant.
  ret = aclnnGroupNormSiluQuantGetWorkspaceSize(self, gamma, beta, quantScale, group, eps, activateSilu, out, meanOut, rstdOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSiluQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second API of aclnnGroupNormSiluQuant.
  ret = aclnnGroupNormSiluQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSiluQuant failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> outResultData(size, 0);
  ret = aclrtMemcpy(outResultData.data(), outResultData.size() * sizeof(outResultData[0]), outDeviceAddr, size * sizeof(int8_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("outResultData[%ld] is: %d\n", i, outResultData[i]);
  }

  // Receive the meanOut result.
  size = GetShapeSize(meanOutShape);
  std::vector<uint16_t> meanResultData(size, 0);
  ret = aclrtMemcpy(meanResultData.data(),
                    meanResultData.size() * sizeof(uint16_t), // 2 bytes
                    meanOutDeviceAddr,
                    size * sizeof(uint16_t), // 2 bytes
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy meanOut from device to host failed. ERROR: %d\n", ret); return ret);

  // Receive the rstdOut result.
  size = GetShapeSize(rstdOutShape);
  std::vector<uint16_t> rstdResultData(size, 0);
  ret = aclrtMemcpy(rstdResultData.data(),
                    rstdResultData.size() * sizeof(uint16_t), // 2 bytes
                    rstdOutDeviceAddr,
                    size * sizeof(uint16_t), // 2 bytes
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy rstdOut from device to host failed. ERROR: %d\n", ret); return ret);

  // Print the result (convert it to float for easy reading).
  for (int64_t i = 0; i < meanResultData.size(); i++) {
    // Simple conversion: Convert the FP16 bit pattern to float.
    __fp16 fp16_val = *reinterpret_cast<__fp16*>(&meanResultData[i]);
    float fp32_val = static_cast<float>(fp16_val);
    LOG_PRINT("meanResultData[%ld] is: %f\n", i, fp32_val);
  }

  for (int64_t i = 0; i < rstdResultData.size(); i++) {
    __fp16 fp16_val = *reinterpret_cast<__fp16*>(&rstdResultData[i]);
    float fp32_val = static_cast<float>(fp16_val);
    LOG_PRINT("rstdResultData[%ld] is: %f\n", i, fp32_val);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(gamma);
  aclDestroyTensor(beta);
  aclDestroyTensor(quantScale);
  aclDestroyTensor(out);
  aclDestroyTensor(meanOut);
  aclDestroyTensor(rstdOut);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(quantScaleDeviceAddr);
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
