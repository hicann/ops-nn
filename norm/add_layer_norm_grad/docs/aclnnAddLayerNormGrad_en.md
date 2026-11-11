# aclnnAddLayerNormGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:35:30.009Z -->

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

- Description: LayerNorm is a normalization method that normalizes the input data of a network layer to a distribution with a mean of 0 and a variance of 1. LayerNormGrad is a key operator in the backward propagation phase of deep learning, mainly used to compute the gradient of the LayerNorm operation. AddLayerNormGrad fuses Add and LayerNormGrad to reduce copy-in and copy-out operations.

- Formula:

  - Forward formula: (D is the size of the reduction axis)

    $$
    x= inputx1 + inputx2
    $$

    $$
    \operatorname{LayerNorm}(x_i)=\frac{x_i−\operatorname{E}(x)}{\sqrt{\operatorname{Var}(x)+ eps}}*gamma + beta
    $$

    $$
    where \operatorname{E}(x)=\frac{1}{D}\sum_{1}^{D}{x_i}
    $$

    $$
    \operatorname{Var}(x)=\frac{1}{D}\sum_{1}^{D}{(x_i-\operatorname{E}(x))^2}
    $$

  - Backward formula:

    $$
    x= inputx1 + inputx2
    $$

    $$
    dxOut = \sum_{i}{inputdy_i * gamma_i * \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}} + dsumOptional
    $$

    $$
    dgammaOut = \sum_{j}{inputdy_i * \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}}
    $$

    $$
    dbetaOut = \sum_{j}{inputdy_i}
    $$

    Where:

    - $\hat{x_j}$:

      $$
      \hat{x_j}=({x_i-\operatorname{E}(x)}) * {rstd}
      $$

    - $rstd$:

      $$
      rstd=\frac {1}{\sqrt{\operatorname{Var}(x)}}
      $$

    - $\frac{{\rm d}\hat{x_j}}{{\rm d}x_i}$:

      $$
      \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}=(\delta_{ij} - \frac{{\rm d}\operatorname{E}(x)}{{\rm d}  x_i}) * \frac{1}{\sqrt{\operatorname{Var}(x)}}-\frac{1}{\operatorname{Var}(x)}  (x_j-\operatorname{E}(x))\frac{\rm d \operatorname{Var}(x)}{\rm dx}
      $$

      Where $\delta_{ij}=1$ when i=j, and $\delta_{ij}=0$ when i!=j.

    - $\frac{{\rm d}\operatorname{E}(x)}{{\rm d}x_i}$:

      $$
      \frac{{\rm d}\operatorname{E}(x)}{{\rm d}x_i}=\frac{1}{D}
      $$

      Where D is the number of elements in x that participate in the mean calculation.

    - $\frac{\rm d \operatorname{Var}(x)}{\rm dx}$:

      $$
      \frac{\rm d \operatorname{Var}(x)}{\rm dx}=\frac{1}{D}\frac{1}{\sqrt{\operatorname{Var} (x)}}(x_i-\operatorname{E}(x))
      $$

    - The simplified $dxOut$:

      $$
      dxOut = rstd * ({inputdy_i * gamma_j} - \frac{1}{D} * (\sum_{j}{inputdy_i * gamma_j} + \hat {x_j} * \sum_{j}{inputdy_i * gamma_j * \hat{x_j}})) + dsumOptional
      $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call "aclnnAddLayerNormGradGetWorkspaceSize" to obtain the workspace size required for calculation and the executor that contains the operator calculation process. Then, call "aclnnAddLayerNormGrad" to perform the calculation.

```Cpp
aclnnStatus aclnnAddLayerNormGradGetWorkspaceSize(
  const aclTensor *dy,
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *rstd,
  const aclTensor *mean,
  const aclTensor *gamma,
  const aclTensor *dsumOptional,
  const aclTensor *dxOut,
  const aclTensor *dgammaOut,
  const aclTensor *dbetaOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddLayerNormGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddLayerNormGradGetWorkspaceSize

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
      <td>dy (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the main grad input. Corresponds to `inputdy` in the formula.</td>
      <td><ul><li>Does not support null Tensor.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x1 (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input x1 of the forward fusion operator. Corresponds to `inputx1` in the formula.</td>
      <td><ul><li>Does not support null tensors.</li><li>The shape and data type must remain consistent with the shape and data type of `dy`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2 (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input x2 of the forward fused operator. Corresponds to `inputx2` in the formula.</td>
      <td><ul><li>Does not support null tensors.</li><li>The shape and data type must remain consistent with the shape and data type of `dy`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the reciprocal of the standard deviation of the sum of forward inputs x1 and x2. This corresponds to `rstd` in the corresponding formula.</td>
      <td><ul><li>Does not support null Tensor.</li><li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with `dy` (the leading dimensions are the same as the leading dimensions of `dy`, where the leading dimensions refer to the dimension of `dy` minus the dimension of `gamma`, indicating the dimensions that do not need norm).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the mean of the sum of the forward inputs x1 and x2. Corresponds to `E(x)` in the formula.</td>
      <td><ul><li>Does not support null tensors.</li><li>The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with `dy` (the leading dimensions are the same as the leading dimensions of `dy`, where leading dimensions refer to the dimension of `dy` minus the dimension of `gamma`, indicating the dimensions that do not need normalization).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the gamma of the forward input. Corresponds to `gamma` in the formula.</td>
      <td><ul><li>Does not support null Tensor.</li><li>The data type must remain consistent with that of `dy`.</li><li>The dimension value of shape is the same as the dimension value of `dy` to be normalized.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dsumOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input. Indicates the additional backward gradient accumulation input. Corresponds to `dsumOptional` in the formula.</td>
      <td><ul><li>Does not support null tensors.</li><li>Shape and data type must remain consistent with the shape and data type of `dy`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dxOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the gradient of `x`, the output of the Add operation. Corresponds to `dxOut` in the formula.</td>
      <td><ul><li>Does not support null tensors.</li><li>Shape and data type must remain consistent with the shape and data type of `dy`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dgammaOut(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the gradient of input parameter `gamma`. It corresponds to `dgammaOut` in the formula.</td>
      <td><ul><li>Does not support null Tensor.</li><li>The shape must be the same as that of input `gamma`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dbetaOut(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the backward gradient of the forward input parameter `beta`, corresponding to `dbetaOut` in the formula.</td>
      <td><ul><li>Does not support null tensors.</li><li>The shape must be consistent with the input `gamma`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize(uint64_t*)</td>
      <td>Output</td>
      <td>Returns the Device side-applied workspace size.</td>
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
  </tbody>
  </table>

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>: The data type of parameters `dy`, `x1`, `x2`, `gamma`, `dsumOptional`, and `dxOut` does not support BFLOAT16.
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
      <td>If the passed parameter is a required input, output, or required attribute and is a null pointer, 161001 is returned.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of a required input or required output is out of the supported range.</td>
    </tr>
  </tbody></table>

## aclnnAddLayerNormGrad

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
      <td>Device side-applied workspace memory address.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Device side-applied workspace size, obtained through the first-phase API aclnnAddLayerNormGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Functional Dimensions**
  - Supported data types

    <!-- npu="310p" id8 -->
    - <term>Atlas inference products</term>: dy, x1, x2, gamma, dsumOptional, and dxOut support FLOAT32 and FLOAT16.
    <!-- end id8 -->
    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: dy, x1, x2, gamma, dsumOptional, and dxOut support FLOAT32, FLOAT16, and BFLOAT16.
    <!-- end id9 -->
    - rstd, mean, dgammaOut, and dbetaOut support: FLOAT32.
  - Data format support: ND.
- **Unsupported Type Description**

  DOUBLE: The instruction does not support DOUBLE.

- **Boundary Value Scenario Description**
  - When the input is Inf, the output is Inf.
  - When the input is NaN, the output is NaN.
- Deterministic computation:
  - aclnnAddLayerNormGrad defaults to a non-deterministic implementation. Deterministic computation can be enabled by calling aclrtCtxSetSysParamOpt.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_layer_norm_grad.h"

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
    // Boilerplate, resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
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
    // 1. (Boilerplate) Initialize device/stream, see the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Process check as needed.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs, customizing the construction according to the API.
    std::vector<int64_t> dyShape = {3, 1, 4};
    std::vector<int64_t> x1Shape = {3, 1, 4};
    std::vector<int64_t> x2Shape = {3, 1, 4};
    std::vector<int64_t> rstdShape = {3, 1, 1};
    std::vector<int64_t> meanShape = {3, 1, 1};
    std::vector<int64_t> gammaShape = {4};
    std::vector<int64_t> dsumOptionalShape = {3, 1, 4};
    std::vector<int64_t> outputpdxShape = {3, 1, 4};
    std::vector<int64_t> outputpdgammaShape = {4};
    std::vector<int64_t> outputpdbetaShape = {4};
    void* dyDeviceAddr = nullptr;
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* dsumOptionalDeviceAddr = nullptr;
    void* outputpdxDeviceAddr = nullptr;
    void* outputpdgammaDeviceAddr = nullptr;
    void* outputpdbetaDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* dsumOptional = nullptr;
    aclTensor* outputpdx = nullptr;
    aclTensor* outputpdgamma = nullptr;
    aclTensor* outputpdbeta = nullptr;
    std::vector<float> dyHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> x1HostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> x2HostData = {2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8};
    std::vector<int32_t> rstdHostData = {0, 1, 2};
    std::vector<int32_t> meanHostData = {0, 1, 2};
    std::vector<int32_t> gammaHostData = {0, 1, 2, 3};
    std::vector<int32_t> dsumOptionalHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> outputpdxHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> outputpdgammaHostData = {0, 1, 2, 3};
    std::vector<int32_t> outputpdbetaHostData = {0, 1, 2, 3};

    // Create the self aclTensor.
    ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        dsumOptionalHostData, dsumOptionalShape, &dsumOptionalDeviceAddr, aclDataType::ACL_FLOAT, &dsumOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outputpdxHostData, outputpdxShape, &outputpdxDeviceAddr, aclDataType::ACL_FLOAT, &outputpdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputpdgammaHostData, outputpdgammaShape, &outputpdgammaDeviceAddr, aclDataType::ACL_FLOAT, &outputpdgamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputpdbetaHostData, outputpdbetaShape, &outputpdbetaDeviceAddr, aclDataType::ACL_FLOAT, &outputpdbeta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnAddLayerNormGrad API call example
    // 3. Call the CANN operator library API. Change it to the specific API name.
    // Call the first-phase API of aclnnAddLayerNormGrad.
    LOG_PRINT("\nUse aclnnAddLayerNormGrad Port.");
    ret = aclnnAddLayerNormGradGetWorkspaceSize(
        dy, x1, x2, rstd, mean, gamma, dsumOptional, outputpdx, outputpdgamma, outputpdbeta, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnAddLayerNormGrad.
    ret = aclnnAddLayerNormGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGrad failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result in device memory to the host. Modify it based on the specific API definition.
    auto outputpdxsize = GetShapeSize(outputpdxShape);
    std::vector<float> resultDataPdx(outputpdxsize, 0);
    ret = aclrtMemcpy(
        resultDataPdx.data(), resultDataPdx.size() * sizeof(resultDataPdx[0]), outputpdxDeviceAddr,
        outputpdxsize * sizeof(resultDataPdx[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdx output");
    for (int64_t i = 0; i < outputpdxsize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdx[i]);
    }
    auto outputpdgammasize = GetShapeSize(outputpdgammaShape);
    std::vector<float> resultDataPdGamma(outputpdgammasize, 0);
    ret = aclrtMemcpy(
        resultDataPdGamma.data(), resultDataPdGamma.size() * sizeof(resultDataPdGamma[0]), outputpdgammaDeviceAddr,
        outputpdgammasize * sizeof(resultDataPdGamma[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdgamma output");
    for (int64_t i = 0; i < outputpdgammasize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdGamma[i]);
    }
    auto outputpdbetasize = GetShapeSize(outputpdbetaShape);
    std::vector<float> resultDataPdBeta(outputpdbetasize, 0);
    ret = aclrtMemcpy(
        resultDataPdBeta.data(), resultDataPdBeta.size() * sizeof(resultDataPdBeta[0]), outputpdbetaDeviceAddr,
        outputpdbetasize * sizeof(resultDataPdBeta[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdbeta output");
    for (int64_t i = 0; i < outputpdbetasize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdBeta[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
    aclDestroyTensor(dy);
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(rstd);
    aclDestroyTensor(mean);
    aclDestroyTensor(gamma);
    aclDestroyTensor(dsumOptional);
    aclDestroyTensor(outputpdx);
    aclDestroyTensor(outputpdgamma);
    aclDestroyTensor(outputpdbeta);

    // 7. Release device resources. Modify it based on the specific API definition.
    aclrtFree(dyDeviceAddr);
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(dsumOptionalDeviceAddr);
    aclrtFree(outputpdxDeviceAddr);
    aclrtFree(outputpdgammaDeviceAddr);
    aclrtFree(outputpdbetaDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
