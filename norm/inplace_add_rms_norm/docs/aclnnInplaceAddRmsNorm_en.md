# aclnnInplaceAddRmsNorm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:11:28.537Z -->

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: The RMSNorm operator is a commonly used normalization operation in large models. Compared with the LayerNorm operator, it removes the part of subtracting the mean. The AddRmsNorm operator fuses the Add operator before RMSNorm to reduce data movement.
- Formula:

  $$
  x_i=x1Ref_i+x2Ref_i
  $$

  $$
  \operatorname{RmsNorm}(x_i)=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * gamma_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

## Function Prototype

Each operator is split into a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the `aclnnInplaceAddRmsNormGetWorkspaceSize` API first to obtain the workspace size required for computation and the executor that contains the computation process. Then call the `aclnnInplaceAddRmsNorm` API to perform the computation.

```Cpp
aclnnStatus aclnnInplaceAddRmsNormGetWorkspaceSize(
  aclTensor       *x1Ref,
  aclTensor       *x2Ref,
  const aclTensor *gamma,
  double           epsilon,
  const aclTensor *rstdOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnInplaceAddRmsNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnInplaceAddRmsNormGetWorkspaceSize

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
      <td>x1Ref(aclTensor*)</td>
      <td>Input/Output</td>
      <td>Represents the first input for the Add computation and the final output after normalization. Correspond to `x1Ref` and `RmsNorm(x)` in the formula respectively.</td>
      <td>Does not support empty tensor.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2Ref (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Represents the second input for the Add computation and the result of the Add computation. Correspond to `x2Ref` and `x` in the formula respectively.</td>
      <td><ul><li>Does not support empty Tensor.</li><li>The shape must remain consistent with `x1Ref`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Scaling factor (weight) of RmsNorm. Corresponds to `gamma` in the formula.</td>
      <td><ul><li>Does not support empty Tensor.</li><li>The shape must remain consistent with the last few dimensions of `x1Ref`. These dimensions are the dimensions of `x1Ref` to be normed.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Value added to the denominator to ensure numerical stability. It is `epsilon` in the corresponding formula.</td>
      <td>The recommended value is 1e-6.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rstdOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the standard deviation after normalization. `Rms(x)` in the corresponding formula.</td>
      <td><ul><li>Does not support empty tensor.</li><li>The number of dimensions remains consistent with that of `x1Ref`, where the dimensions that do not need normalization (the first few dimensions obtained by subtracting the dimensions of `gamma` from those of `x1Ref`) remain consistent with the corresponding dimensions of `x1Ref`, and the dimensions that need normalization (the last few dimensions with the same number of dimensions as `gamma`) are all 1. Examples of the relationship between the shape of `rstdOut` and the shapes of `x1Ref` and `gamma`: If the shape of `x1Ref` is (2,3,4,8) and the shape of `gamma` is (8), the shape of `rstdOut` is (2,3,4,1). If the shape of `x1Ref` is (2,3,4,8) and the shape of `gamma` is (4,8), the shape of `rstdOut` is (2,3,1,1).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size required to be side-applied on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor. It contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>:
    - The data type of the parameters `x1Ref`, `x2Ref`, and `gamma` does not support BFLOAT16.
    - The parameter `rstdOut` is invalid in the current product usage scenario.
  <!-- end id7 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
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
      <td>If a parameter is a required input, output, or required attribute and is a null pointer, 161001 is returned.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of the input or output is not in the supported range.</td>
    </tr>
  </tbody></table>

## aclnnInplaceAddRmsNorm

- **Parameters**

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
      <td>Workspace memory size allocated on the Device side. The value is obtained by calling the first-phase API aclnnInplaceAddRmsNormGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. (For details, see [aclnn Error Codes](../../../docs/en/context/aclnn_return_code.md).)

## Constraints

- Boundary value scenario description:
  - When the input is Inf, the output is Inf.
  - When the input is NaN, the output is NaN.
- Deterministic computation:
  - aclnnInplaceAddRmsNorm defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_inplace_add_rms_norm.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate: resource initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId); aclFinalize(); return ret);

    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the Device side.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data from the host side to the Device side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs. Custom construction is required based on the API.
    std::vector<int64_t> xShape = {2, 16};
    std::vector<int64_t> gammaShape = {16};
    std::vector<int64_t> rstdShape = {2, 1};
    void* x1RefDeviceAddr = nullptr;
    void* x2RefDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    aclTensor* x1Ref = nullptr;
    aclTensor* x2Ref = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* rstd = nullptr;
    std::vector<float> x1RefHostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                                        0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> x2RefHostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                                        0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> gammaHostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> rstdHostData = {1, 2};
    float epsilon = 1e-6;

    // Create x1Ref aclTensor.
    ret = CreateAclTensor(x1RefHostData, xShape, &x1RefDeviceAddr, aclDataType::ACL_FLOAT, &x1Ref);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create x2Ref aclTensor.
    ret = CreateAclTensor(x2RefHostData, xShape, &x2RefDeviceAddr, aclDataType::ACL_FLOAT, &x2Ref);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gamma aclTensor.
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the rstd aclTensor.
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the aclnnInplaceAddRmsNorm first-phase API.
    ret = aclnnInplaceAddRmsNormGetWorkspaceSize(x1Ref, x2Ref, gamma, epsilon, rstd, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the aclnnInplaceAddRmsNorm second-phase API.
    ret = aclnnInplaceAddRmsNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddRmsNorm failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device side to the host side. Modify it based on the specific API definition.
    auto size = GetShapeSize(xShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), x1RefDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("x1Ref result[%ld] is: %f\n", i, resultData[i]);
    }

    std::vector<float> resultData1(size, 0);
    ret = aclrtMemcpy(
        resultData1.data(), resultData1.size() * sizeof(resultData1[0]), x2RefDeviceAddr, size * sizeof(resultData1[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("x2Ref result[%ld] is: %f\n", i, resultData1[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify them based on the specific API definition.
    aclDestroyTensor(x1Ref);
    aclDestroyTensor(x2Ref);
    aclDestroyTensor(gamma);
    aclDestroyTensor(rstd);

    // 7. Release device resources. Modify them based on the specific API definition.
    aclrtFree(x1RefDeviceAddr);
    aclrtFree(x2RefDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
