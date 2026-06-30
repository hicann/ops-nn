# aclnnDeepNormGrad

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/deep_norm_grad)

## Product Support

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     √    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description: Computes the backward pass of [aclnnDeepNorm](../../deep_norm/docs/aclnnDeepNorm_en.md), calculating the gradients of tensors **x**, **gx**, and **gamma**, as well as the sum of tensor **dy**.

- Formula:
  
  $$
  dgx_i = tmpone_i * rstd + dvar * tmptwo_i + dmean
  $$
  
  $$
  dx_i = alpha * {dgx}_i
  $$
  
  $$
  dbeta = \sum_{i=1}^{N} dy_i
  $$
  
  $$
  dgamma =  \sum_{i=1}^{N} dy_i * rstd * {tmptwo}_i
  $$
  
  Where:
  
  $$
  oneDiv=-1/SizeOf(gamma)
  $$
  
  $$
  tmpone_i = dy_i * gamma
  $$
  
  $$
  tmptwo_i = alpha * x_i + {gx}_i - mean
  $$
  
  $$
  dvar = (oneDiv) * \sum_{i=1}^{N} {tmpone}_i * {tmptwo}_i * {rstd}^3
  $$
  
  $$
  dmean = (oneDiv) * \sum_{i=1}^{N} {tmpone}_i * rstd
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnDeepNormGradGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnDeepNormGrad** is called to perform computation.

```Cpp
aclnnStatus aclnnDeepNormGradGetWorkspaceSize(
  const aclTensor *dy,
  const aclTensor *x,
  const aclTensor *gx,
  const aclTensor *gamma,
  const aclTensor *mean,
  const aclTensor *rstd,
  double           alpha,
  const aclTensor *dxOut,
  const aclTensor *dgxOut,
  const aclTensor *dbetaOut,
  const aclTensor *dgammaOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDeepNormGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDeepNormGradGetWorkspaceSize

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
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>Primary gradient input, corresponding to `dy` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type and shape must be the same as those of `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input tensor of the forward fused operator, corresponding to `x` in the formula.</td>
      <td>Empty tensors are supported.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gx</td>
      <td>Input</td>
      <td>Input tensor of the forward fused operator, corresponding to `gx` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type and shape must be the same as those of <idp:inline displayname="code" id="code11204141814456">x</idp:inline>.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Scaling parameter from the forward pass, corresponding to `gamma` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of `x`. </li><li>The shape dimensions must match the trailing dimensions of `x` (the dimensions to be normalized).</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1–7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Mean of the sum of forward inputs x and gx, corresponding to `mean` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape dimensions must match the leading dimensions of `x` (the dimensions not to be normalized).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Input</td>
      <td>Reciprocal standard deviation of the sum of forward inputs x and gx, corresponding to `rstd` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as that of `mean`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Weight parameter for adjusting the input data, corresponding to `alpha` in the formula.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dxOut</td>
      <td>Output</td>
      <td>Computed gradient for updating the input tensor x, corresponding to `dx` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type and shape must be the same as those of <idp:inline displayname="code" id="code1206918104511">x</idp:inline>.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dgxOut</td>
      <td>Output</td>
      <td>Computed gradient for updating the input tensor gx, corresponding to `dgx` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type and shape must be the same as those of <idp:inline displayname="code" id="code142077184451">x</idp:inline>.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dbetaOut</td>
      <td>Output</td>
      <td>Computed gradient for updating the bias parameter, corresponding to `dbeta` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as that of <idp:inline displayname="code" id="code6282181513555">gamma</idp:inline>.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1–7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dgammaOut</td>
      <td>Output</td>
      <td>Computed gradient for updating the scaling parameter, corresponding to `dgamma` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as that of `gamma`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1–7</td>
      <td>√</td>
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
      <td>Operator executor, containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - For <term>Atlas inference series products</term>, the data types of `dy`, `x`, `gx`, `gamma`, `dxOut`, and `dgxOut` cannot be BFLOAT16.

- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
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
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The input or output data type is not supported.</td>
    </tr>
    <tr>
      <td>The input and output shapes do not match or are unsupported.</td>
    </tr>
    <tr>
      <td>The input and output data types do not meet the constraints specified in the parameter description.</td>
    </tr>
  </tbody></table>

## aclnnDeepNormGrad

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnDeepNormGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Unsupported types:

  **DOUBLE**: not supported by the instruction set.
- Boundary value specifications:
  * The output is **Inf** when the input is **Inf**.
  * The output is **NaN** when the input is **NaN**.
- Deterministic computation:
  - **aclnnDeepNormGrad** defaults to a non-deterministic implementation. Enabling deterministic computation via **aclrtCtxSetSysParamOpt** is not supported.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_deep_norm_grad.h"

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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
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
    // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Customize error handling based on your requirements.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs based on API definitions.
    float alpha = 0.3;
    std::vector<int64_t> dyShape = {3, 1, 4};
    std::vector<int64_t> xShape = {3, 1, 4};
    std::vector<int64_t> gxShape = {3, 1, 4};
    std::vector<int64_t> gammaShape = {4};
    std::vector<int64_t> meanShape = {3, 1, 1};
    std::vector<int64_t> rstdShape = {3, 1, 1};
    std::vector<int64_t> outputpdxShape = {3, 1, 4};
    std::vector<int64_t> outputpdgxShape = {3, 1, 4};
    std::vector<int64_t> outputpdbetaShape = {4};
    std::vector<int64_t> outputpdgammaShape = {4};
    void* dyDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* gxDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* outputpdxDeviceAddr = nullptr;
    void* outputpdgxDeviceAddr = nullptr;
    void* outputpdbetaDeviceAddr = nullptr;
    void* outputpdgammaDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* x = nullptr;
    aclTensor* gx = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* outputpdx = nullptr;
    aclTensor* outputpdgx = nullptr;
    aclTensor* outputpdbeta = nullptr;
    aclTensor* outputpdgamma = nullptr;

    std::vector<float> dyHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> xHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> gxHostData = {2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8};
    std::vector<float> gammaHostData = {0, 1, 2, 3};
    std::vector<float> meanHostData = {0, 1, 2};
    std::vector<float> rstdHostData = {0, 1, 2};
    std::vector<float> outputpdxHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> outputpdgxHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> outputpdbetaHostData = {0, 1, 2, 3};
    std::vector<float> outputpdgammaHostData = {0, 1, 2, 3};

    // Create a self aclTensor.
    ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gxHostData, gxShape, &gxDeviceAddr, aclDataType::ACL_FLOAT, &gx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outputpdxHostData, outputpdxShape, &outputpdxDeviceAddr, aclDataType::ACL_FLOAT, &outputpdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputpdgxHostData, outputpdgxShape, &outputpdgxDeviceAddr, aclDataType::ACL_FLOAT, &outputpdgx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputpdbetaHostData, outputpdbetaShape, &outputpdbetaDeviceAddr, aclDataType::ACL_FLOAT, &outputpdbeta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputpdgammaHostData, outputpdgammaShape, &outputpdgammaDeviceAddr, aclDataType::ACL_FLOAT, &outputpdgamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnDeepNormGrad API call example
    // 3. Call the CANN operator library API. Modify the API as required.
    // Call the first-phase API of aclnnDeepNormGrad.
    LOG_PRINT("\nUse aclnnDeepNormGrad Port.");
    ret = aclnnDeepNormGradGetWorkspaceSize(
        dy, x, gx, gamma, mean, rstd, alpha, outputpdx, outputpdgx, outputpdbeta, outputpdgamma, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeepNormGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnDeepNormGrad.
    ret = aclnnDeepNormGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeepNormGrad failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize the stream and wait for task completion.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
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
    auto outputpdgxsize = GetShapeSize(outputpdgxShape);
    std::vector<float> resultDataPdgx(outputpdgxsize, 0);
    ret = aclrtMemcpy(
        resultDataPdgx.data(), resultDataPdgx.size() * sizeof(resultDataPdgx[0]), outputpdgxDeviceAddr,
        outputpdgxsize * sizeof(resultDataPdgx[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdgx output");
    for (int64_t i = 0; i < outputpdgxsize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdgx[i]);
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

    // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(dy);
    aclDestroyTensor(x);
    aclDestroyTensor(gx);
    aclDestroyTensor(gamma);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);
    aclDestroyTensor(outputpdx);
    aclDestroyTensor(outputpdgx);
    aclDestroyTensor(outputpdbeta);
    aclDestroyTensor(outputpdgamma);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(dyDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(gxDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(outputpdxDeviceAddr);
    aclrtFree(outputpdgxDeviceAddr);
    aclrtFree(outputpdbetaDeviceAddr);
    aclrtFree(outputpdgammaDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
