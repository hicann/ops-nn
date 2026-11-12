# aclnnRmsNormGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:33:46.424Z -->

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

- Description: Backward computation of [aclnnRmsNorm](../../rms_norm/docs/aclnnRmsNorm.md). Computes the gradient of RmsNorm, that is, the gradient of the input tensor during backward propagation.
- Operator formula:

  - Forward formula:

  $$
  \operatorname{RmsNorm}(x_i)=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  - Backward derivation:

  $$
  dx_i= (dy_i * g_i - \frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * \operatorname{Mean}(\mathbf{y})) * \frac{1} {\operatorname{Rms}(\mathbf{x})},  \quad \text { where } \operatorname{Mean}(\mathbf{y}) = \frac{1}{n}\sum_{i=1}^n (dy_i * g_i * x_i * \frac{1}{\operatorname{Rms}(\mathbf{x})})
  $$

  $$
  dg_i = \frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * dy_i
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the `aclnnRmsNormGradGetWorkspaceSize` API to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the `aclnnRmsNormGrad` API to perform the computation.

```Cpp
aclnnStatus aclnnRmsNormGradGetWorkspaceSize(
  const aclTensor *dy,
  const aclTensor *x,
  const aclTensor *rstd,
  const aclTensor *gamma,
  const aclTensor *dxOut,
  const aclTensor *dgammaOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnRmsNormGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnRmsNormGradGetWorkspaceSize

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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>dy (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the gradient passed back in backward propagation, corresponding to `dy` in the formula.</td>
      <td>Supports a null tensor.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input of the forward operator, that is, the data to be normalized, corresponding to `x` in the formula.</td>
      <td><ul><li>Supports null Tensor.</li><li>The shape must remain consistent with the shape of input parameter `dy`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the intermediate computation result of the forward operator, corresponding to `1/Rms(x)` in the formula.</td>
      <td><ul><li>Supports null Tensor.</li><li>The shape must satisfy rstd_shape = x_shape[0:n], where n < x_shape.dims() and n remains consistent with the n of `gamma`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Scaling factor (weight) for normalization computation of the forward operator, corresponding to `gamma` in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>The shape must satisfy gamma_shape = x_shape[n:], n < x_shape.dims().</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dxOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the gradient of input `x`, corresponding to `dx` in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>The shape remains consistent with the shape of input parameter `dy`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dgammaOut (aclTensor*)</td>
      <td>Output</td>
      <td>Gradient of `gamma`, corresponding to `dg` in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>The shape remains consistent with the shape of input parameter `gamma`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied on the Device side by the user.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>: The data type of parameters `dy`, `x`, `gamma`, and `dxOut` does not support BFLOAT16.
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
      <td>If a required input, output, or attribute is a null pointer, 161001 is returned.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The input or output data type is not within the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The parameters do not meet the requirements described in the parameter description.</td>
    </tr>
  </tbody></table>

## aclnnRmsNormGrad

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
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnRmsNormGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

<!-- npu="310p" id8 -->
- <term>Atlas inference products</term>: The last-axis length of the `x`, `dy`, and `gamma` inputs must be greater than or equal to 32 Bytes.
<!-- end id8 -->

- Supported data types for each product:

  <!-- npu="950,A3,910b" id9 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:

    | `dy` Data Type | `x` Data Type | `rstd` Data Type | `gamma` Data Type | `dxOut` Data Type | `dgammaOut` Data Type |
    | -------- | -------- | -------- | -------- | -------- | -------- |
    | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT16  | FLOAT32  |
    | BFLOAT16 | BFLOAT16 | FLOAT32  | FLOAT32  | BFLOAT16 | FLOAT32  |
    | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT16  | FLOAT16  | FLOAT32  |
    | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | BFLOAT16 | BFLOAT16 | FLOAT32  | BFLOAT16 | BFLOAT16 | FLOAT32  |
  <!-- end id9 -->

  <!-- npu="310p" id10 -->
  - <term>Atlas inference products</term>:

    | `dy` Data Type | `x` Data Type | `rstd` Data Type | `gamma` Data Type | `dxOut` Data Type | `dgammaOut` Data Type |
    | -------- | -------- | -------- | -------- | -------- | -------- |
    | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT16  | FLOAT16  | FLOAT32  |
    | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
  <!-- end id10 -->

- Deterministic computation:

  <!-- npu="A3,910b,310p" id11 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas inference products</term>: aclnnRmsNormGrad defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtCtxSetSysParamOpt.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnRmsNormGrad defaults to a deterministic implementation.
  <!-- end id12 -->

## Example

The sample code is as follows for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm_grad.h"
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
    // Boilerplate for resource initialization.
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
    // Call aclrtMalloc to allocate memory on the device side.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data from the host side to the device side.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW, shape.data(),
        shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output. Custom construction is required based on the API definition.
    std::vector<int64_t> gradInputShape = {2, 1, 16};
    std::vector<int64_t> xInputShape = {2, 1, 16};
    std::vector<int64_t> rstdInputShape = {2};
    std::vector<int64_t> gammaInputShape = {16};
    std::vector<int64_t> dxOutputShape = {2, 1, 16};
    std::vector<int64_t> dgammaOutputShape = {16};

    void* gradInputDeviceAddr = nullptr;
    void* xInputDeviceAddr = nullptr;
    void* rstdInputDeviceAddr = nullptr;
    void* gammaInputDeviceAddr = nullptr;
    void* dxOutDeviceAddr = nullptr;
    void* dgammaOutDeviceAddr = nullptr;

    aclTensor* gradInput = nullptr;
    aclTensor* xInput = nullptr;
    aclTensor* rstdInput = nullptr;
    aclTensor* gammaInput = nullptr;
    aclTensor* dxOut = nullptr;
    aclTensor* dgammaOut = nullptr;

    std::vector<float> gradInputHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    std::vector<float> xInputHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    std::vector<float> rstdInputHostData = {1, 2};
    std::vector<float> gammaInputHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    std::vector<float> dxOutHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    std::vector<float> dgammaOutHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    std::vector<int64_t> output1SizeData = {2, 1, 16};
    std::vector<int64_t> output2SizeData = {16};
    std::vector<int64_t> input1SizeData = {2, 1, 16};
    std::vector<int64_t> input2SizeData = {2};
    std::vector<int64_t> input3SizeData = {16};

    ret = CreateAclTensor(gradInputHostData, input1SizeData, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(xInputHostData, input1SizeData, &xInputDeviceAddr, aclDataType::ACL_FLOAT, &xInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(rstdInputHostData, input2SizeData, &rstdInputDeviceAddr, aclDataType::ACL_FLOAT, &rstdInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret =
        CreateAclTensor(gammaInputHostData, input3SizeData, &gammaInputDeviceAddr, aclDataType::ACL_FLOAT, &gammaInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(dxOutHostData, output1SizeData, &dxOutDeviceAddr, aclDataType::ACL_FLOAT, &dxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(dgammaOutHostData, output2SizeData, &dgammaOutDeviceAddr, aclDataType::ACL_FLOAT, &dgammaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnRmsNormGrad.
    ret = aclnnRmsNormGradGetWorkspaceSize(
        gradInput, xInput, rstdInput, gammaInput, dxOut, dgammaOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnRmsNormGrad.
    ret = aclnnRmsNormGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormGrad failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device side to the host side. Modify it based on the specific API definition.
    auto size_dx = GetShapeSize(gradInputShape);
    std::vector<float> resultData1(size_dx, 0);
    ret = aclrtMemcpy(
        resultData1.data(), resultData1.size() * sizeof(resultData1[0]), dxOutDeviceAddr, size_dx * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size_dx; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData1[i]);
    }
    auto size_dgamma = GetShapeSize(gammaInputShape);
    std::vector<float> resultData2(size_dgamma, 1);
    ret = aclrtMemcpy(
        resultData2.data(), resultData2.size() * sizeof(resultData2[0]), dgammaOutDeviceAddr,
        size_dgamma * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size_dgamma; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData2[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
    aclDestroyTensor(gradInput);
    aclDestroyTensor(xInput);
    aclDestroyTensor(rstdInput);
    aclDestroyTensor(gammaInput);
    aclDestroyTensor(dxOut);
    aclDestroyTensor(dgammaOut);

    // 7. Release device resources. Modify the code based on the specific API definition.
    aclrtFree(gradInputDeviceAddr);
    aclrtFree(xInputDeviceAddr);
    aclrtFree(rstdInputDeviceAddr);
    aclrtFree(gammaInputDeviceAddr);
    aclrtFree(dxOutDeviceAddr);
    aclrtFree(dgammaOutDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
