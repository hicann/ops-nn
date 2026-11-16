# aclnnFastLayerNorm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:19:39.083Z -->

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

- Description: Performs normalization calculation on the specified layer with a mean of 0 and a standard deviation of 1. Compared with the aclnnLayerNorm API, the aclnnFastLayerNorm API improves overall performance by 50%, with memory usage consistent with that of the GPU. Differences in precision exist due to the optimized accumulation order.

- Formula:

  $$
  out = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + eps}} * weightOptional + biasOptional
  $$

  $$
  meanOutOptional = \mathrm{E}[x]
  $$

  $$
  rstdOutOptional = \frac{1}{ \sqrt{\mathrm{Var}[x] + eps}}
  $$

  Where E[x] represents the mean of the input, and Var[x] represents the variance of the input.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnFastLayerNormGetWorkspaceSize" API to obtain the workspace size required for calculation and the executor that includes the operator calculation process, and then call the "aclnnFastLayerNorm" API to execute the calculation.

```Cpp
aclnnStatus aclnnFastLayerNormGetWorkspaceSize(
  const aclTensor   *input,
  const aclIntArray *normalizedShape,
  const aclTensor   *weightOptional,
  const aclTensor   *biasOptional,
  double             eps,
  aclTensor         *out,
  aclTensor         *meanOutOptional,
  aclTensor         *rstdOutOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnFastLayerNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnFastLayerNormGetWorkspaceSize

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
      <td>input (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input for the normalization calculation, corresponding to `x` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is [A1,...,Ai,R1,...,Rj], where A1 to Ai represent dimensions that do not require norm, and R1 to Rj represent dimensions that require norm.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>normalizedShape (aclIntArray*)</td>
      <td>Input</td>
      <td>Indicates the dimensions for norm calculation.</td>
      <td>The value is [R1,...,Rj]. Its length is less than or equal to the shape length of input, and empty is not supported. R1*R2*...*Rj is less than or equal to 583705600.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input parameter, indicating the weight used for the normalization calculation. It corresponds to `weightOptional` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>When `weightOptional` is non-empty:<ul><li>The data type is the same as that of `input` or is FLOAT. When `biasOptional` is present, `weightOptional` and `biasOptional` have the same data type.</li><li>The shape is equal to `normalizedShape`, which is [R1,...,Rj].</li></ul></li><li>When `weightOptional` is nullptr, the calculation uses the default value 1.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input parameter. Indicates the offset for performing the normalization calculation. Corresponds to `biasOptional` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>When `biasOptional` is non-empty:<ul><li>The data type is the same as that of input `input` or is FLOAT. When `weightOptional` is present, the data type of `biasOptional` is the same as that of `weightOptional`.</li><li>The shape is equal to `normalizedShape`, which is [R1,...,Rj].</li></ul></li><li>When `biasOptional` is nullptr, the calculation uses the default value 0.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>eps (double)</td>
      <td>Input</td>
      <td>Indicates the value added to the denominator to ensure numerical stability. Corresponds to `eps` in the corresponding formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the result of the normalization calculation. It corresponds to `out` in the corresponding formula.</td>
      <td><ul><li>Supports empty Tensor.</li><li>The shape must be equal to the shape of `input`, which is [A1,...,Ai,R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>meanOutOptional (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output, indicating the mean after normalization. It corresponds to `meanOutOptional` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>The shape is [A1,...,Ai,1,...,1], with a total of j ones after Ai, and the number of ones must be equal to the length of the axis to be normalized. When `rstdOutOptional` is present, the shape is the same as that of `rstdOutOptional`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOutOptional (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output, indicating the reciprocal of the standard deviation after performing normalization. It corresponds to `rstdOutOptional` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>The shape is [A1,...,Ai,1,...,1], with a total of j ones after Ai, and the number of ones must be equal to the length of the axis to be normalized. When `meanOutOptional` is present, the shape is the same as that of `meanOutOptional`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace that needs to be applied on the side of Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which includes calculation process of the operator.</td>
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
      <td>The input, normalizedShape, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>The dimension of shape in input, weightOptional (when non-empty), biasOptional (when non-empty), out, meanOutOptional (when non-empty), or rstdOutOptional (when non-empty) exceeds 8.</td>
    </tr>
    <tr>
      <td>The number of elements in normalizedShape exceeds 8.</td>
    </tr>
    <tr>
      <td>The data type of input, weightOptional (when non-empty), biasOptional (when non-empty), out, meanOutOptional (when non-empty), or rstdOutOptional (when non-empty) is not supported.</td>
    </tr>
    <tr>
      <td>The dimension of normalizedShape is less than 1.</td>
    </tr>
    <tr>
      <td>weightOptional is non-empty and its shape is not equal to the shape of normalizedShape.</td>
    </tr>
    <tr>
      <td>biasOptional is non-empty and its shape is not equal to the shape of normalizedShape.</td>
    </tr>
    <tr>
      <td>The dimension of input is less than the dimension of normalizedShape.</td>
    </tr>
    <tr>
      <td>The shapes of corresponding dimensions are not equal when the shape of input is right-aligned with the shape of normalizedShape.</td>
    </tr>
    <tr>
      <td>The shape of input is inconsistent with that of out.</td>
    </tr>
    <tr>
      <td>The shape of meanOutOptional (when non-empty) or rstdOutOptional (when non-empty) is not [A1,...,Ai,1,...,1].</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>When the value of normalizedShape is [R1,...,Rj], the value of R1*R2*...*Rj is greater than 583705600.</td>
    </tr>
  </tbody></table>

## aclnnFastLayerNorm

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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnFastLayerNormGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, includes calculation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream that executes tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

<!-- npu="A3,910b" id7 -->
- <term>Atlas A3 series products</term> and <term>Atlas A2 series products</term>: The data type of `out` must be the same as that of `input`.
<!-- end id7 -->

- The shape of input, normalizedShape, weightOptional (when non-null), biasOptional (when non-null), out, meanOutOptional (when non-null), or rstdOutOptional (when non-null) does not exceed 8 dimensions.
- deterministic computation:
  - aclnnFastLayerNorm defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_layer_norm.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // boilerplate, resource initialization
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
int CreateAclTensorMem(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
void aclCreateTensorP(const std::vector<T>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
}

template <typename T>
void aclCreateIntArrayP(const std::vector<T>& hostData, aclIntArray** intArray)
{
    *intArray = aclCreateIntArray(hostData.data(), hostData.size());
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Customize the construction based on the API definitions.
    std::vector<int64_t> xShape = {1, 2, 32};
    std::vector<int64_t> normShape = {32};
    std::vector<int64_t> meanShape = {1, 2, 1};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclIntArray* norm = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* out = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* rstd = nullptr;
    std::vector<float> xHostData(64, 2.0);
    std::vector<int64_t> normData = {32};
    std::vector<float> weightHostData(32, 1.0);
    std::vector<float> biasHostData(32, 0.0);
    std::vector<float> outHostData(64, 0.0);
    std::vector<float> meanHostData(2, 0.0);
    std::vector<float> rstdHostData(2, 0.0);
    double eps = 1e-5;

    ret = CreateAclTensorMem(xHostData, xShape, &xDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorMem(weightHostData, normShape, &weightDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorMem(biasHostData, normShape, &biasDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorMem(outHostData, xShape, &outDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorMem(meanHostData, meanShape, &meanDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorMem(rstdHostData, meanShape, &rstdDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclCreateTensorP(xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    aclCreateIntArrayP(normData, &norm);
    aclCreateTensorP(normShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    aclCreateTensorP(normShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    aclCreateTensorP(xShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    aclCreateTensorP(meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    aclCreateTensorP(meanShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnFastLayerNorm.
    ret = aclnnFastLayerNormGetWorkspaceSize(x, norm, weight, bias, eps, out, mean, rstd, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFastLayerNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnFastLayerNorm.
    ret = aclnnFastLayerNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFastLayerNorm failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronously wait for the task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value. Copy the result from device memory to host memory. Modify it based on the specific API definition.
    auto size = GetShapeSize(xShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy first result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out result[%ld] is: %f\n", i, resultData[i]);
    }

    auto size1 = GetShapeSize(meanShape);
    std::vector<float> resultData1(size1, 0);
    ret = aclrtMemcpy(
        resultData1.data(), resultData1.size() * sizeof(resultData1[0]), meanDeviceAddr, size1 * sizeof(resultData1[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy second result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("mean result[%ld] is: %f\n", i, resultData1[i]);
    }

    auto size2 = GetShapeSize(meanShape);
    std::vector<float> resultData2(size2, 0);
    ret = aclrtMemcpy(
        resultData2.data(), resultData2.size() * sizeof(resultData2[0]), rstdDeviceAddr, size2 * sizeof(resultData2[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy last result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("rstd result[%ld] is: %f\n", i, resultData2[i]);
    }

    // 6. Release aclTensor and aclIntArray. Modify it based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyIntArray(norm);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(out);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);

    // 7. Release device resources.
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(meanDeviceAddr);
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
