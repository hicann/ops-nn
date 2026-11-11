# aclnnRmsNorm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:28:28.939Z -->

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

- Description: The RmsNorm operator is a normalization operation commonly used in large models. Compared with the LayerNorm operator, it removes the part of subtracting the mean.
- Formula:

  $$
  \operatorname{RmsNorm}(x_i)=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * gamma_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call `aclnnRmsNormGetWorkspaceSize` first to obtain the workspace size required for the computation and an executor that contains the operator computation process, and then call `aclnnRmsNorm` to execute the computation.

```Cpp
aclnnStatus aclnnRmsNormGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *gamma,
  double           epsilon,
  const aclTensor *yOut,
  const aclTensor *rstdOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnRmsNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnRmsNormGetWorkspaceSize

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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input for normalization computation, corresponding to `x` in the formula.</td>
      <td>Supports empty tensors.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Scaling factor (weight) for normalization computation, corresponding to `gamma` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape must satisfy gamma_shape = x_shape[n:], n &lt; x_shape.dims().</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Value added to the variance to avoid division by zero. Corresponds to `epsilon` in the corresponding formula.</td>
      <td>Default value: 1e-6.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Final output after normalization, corresponding to `RmsNorm(x)` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is the same as the shape of input `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the reciprocal of the root mean square (RMS) after normalization, used for the normalization operation. It corresponds to the reciprocal of `Rms(x)` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>Has the same number of dimensions as `x`. Dimensions that do not need normalization are the same as the corresponding dimensions of `x`, and dimensions that need normalization (the last few dimensions, whose number is the same as the number of dimensions of `gamma`) are all 1. Example of the relationship between the rstdOut shape and the x and gamma shapes: If x shape is (2,3,4,8) and gamma shape is (8), rstdOut shape is (2,3,4,1); if x shape is (2,3,4,8) and gamma shape is (4,8), rstdOut shape is (2,3,1,1).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>: The data types of parameters `x`, `gamma`, and `yOut` do not support BFLOAT16.
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
      <td>If a required input, output, or required attribute is a null pointer, 161001 is returned.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of the input or output is out of the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The parameters do not meet the requirements described in Parameters.</td>
    </tr>
  </tbody></table>

## aclnnRmsNorm

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnRmsNormGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream where tasks are executed.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

<!-- npu="310p" id8 -->
- <term>Atlas inference products</term>: The length of the last dimension of the x and gamma inputs must be greater than or equal to 32 bytes.
<!-- end id8 -->
- Boundary value scenario description:

  <!-- npu="310p" id9 -->
  - <term>Atlas inference products</term>: The input does not support containing Inf and NaN.
  <!-- end id9 -->
  <!-- npu="950,A3,910b" id10 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: When the input is Inf, the output is Inf. When the input is NaN, the output is NaN.
  <!-- end id10 -->

- Supported data types by platform:

  <!-- npu="950,A3,910b" id11 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:

    | Data Type Of `x` | `gamma` Data Type | `yOut` Data Type | `rstdOut` Data Type |
    | -------- | -------- | -------- | -------- |
    | FLOAT16 | FLOAT32  | FLOAT16 | FLOAT32 |
    | BFLOAT16 | FLOAT32 | BFLOAT16 | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 |
    | FLOAT32 | FLOAT32  | FLOAT32 | FLOAT32  |
  <!-- end id11 -->

  <!-- npu="310p" id12 -->
  - <term>Atlas inference products</term>:

    | Data Type Of `x` | `gamma` Data Type | `yOut` Data Type | `rstdOut` Data Type |
    | -------- | -------- | -------- | -------- |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 |
    | FLOAT32 | FLOAT32  | FLOAT32 | FLOAT32  |
  <!-- end id12 -->

- Deterministic computation:
  - aclnnRmsNorm defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm.h"

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
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Device/stream initialization. For details, see the acl API reference.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Process check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs as required by the API.
    std::vector<int64_t> xShape = {2, 16};
    std::vector<int64_t> gammaShape = {16};
    std::vector<int64_t> yShape = {2, 16};
    std::vector<int64_t> rstdShape = {2, 1};
    void* xDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* y = nullptr;
    aclTensor* rstd = nullptr;
    std::vector<float> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                                    0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> gammaHostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> yHostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                                    0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> rstdHostData = {1, 2};
    double epsilon = 1e-6;

    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gamma aclTensor.
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the y aclTensor.
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the rstd aclTensor.
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, and replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnRmsNorm.
    ret = aclnnRmsNormGetWorkspaceSize(x, gamma, epsilon, y, rstd, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnRmsNorm.
    ret = aclnnRmsNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNorm failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait synchronously for the task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value, copy the result from the device memory to the host, and modify based on the specific API definition.
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar, and modify based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(y);
    aclDestroyTensor(rstd);

    // 7. Release device resources, and modify based on the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(yDeviceAddr);
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
