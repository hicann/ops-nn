# aclnnAddLayerNormGrad

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/add_layer_norm_grad)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     √    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description: Normalizes the input data at the network layer to the [0, 1] range. The LayerNormGrad operator is a key operator used in the backpropagation phase of deep learning. It is mainly used to compute the gradient of the LayerNorm operation. The AddLayerNormGrad operator fuses the Add and LayerNormGrad operators to reduce data move-in and move-out operations.

- Formula:

  - Forward formula: (D indicates the size of the reduced axis.)

    $$
    x= inputx1 + inputx2
    $$

    $$
    \operatorname{LayerNorm}(x)=\frac{x_i−\operatorname{E}(x)}{\sqrt{\operatorname{Var}(x)+ eps}}*gamma + beta
    $$

    $$
    \operatorname{E}(x_i)=\frac{1}{D}\sum_{1}^{D}{x_i}
    $$

    $$
    \operatorname{Var}(x_i)=\frac{1}{D}\sum_{1}^{D}{(x_i-\operatorname{E}(x))^2}
    $$

  - Backward formula:

    $$
    x= inputx1 + inputx2
    $$

    $$
    dxOut = \sum_{j}{inputdy_i * gamma_j * \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}} + dsumOptional
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
      \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}=(\delta_{ij} - \frac{{\rm d}\operatorname{E}(x)}{{\rm d}  x_i}) * \frac{1}{\sqrt{\operatorname{Var}(x_i)}}-\frac{1}{\operatorname{Var}(x_i)}  (x_j-\operatorname{E}(x))\frac{\rm d \operatorname{Var}(x_i)}{\rm dx}
      $$

      When i=j, $\delta_{ij}$=1; when i!=j, $\delta_{ij}$=0.


    - $\frac{{\rm d}\operatorname{E}(x)}{{\rm d}x_i}$:

      $$
      \frac{{\rm d}\operatorname{E}(x)}{{\rm d}x_i}=\frac{1}{D}
      $$

      D is the number of elements in x that participate in the mean calculation.

    - $\frac{\rm d \operatorname{Var}(x_i)}{\rm dx}$:

      $$
      \frac{\rm d \operatorname{Var}(x_i)}{\rm dx}=\frac{1}{D}\frac{1}{\sqrt{\operatorname{Var}  (x_i)}}(x_i-\operatorname{E}(x))
      $$

    - Simplified $dxOut$:

      $$
      dxOut = rstd * ({inputdy_i * gamma_j} - \frac{1}{D} * (\sum_{j}{inputdy_i * gamma_j} + \hat      {x_j} * \sum_{j}{inputdy_i * gamma_j * \hat{x_j}})) + dsumOptional
      $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnAddLayerNormGradGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnAddLayerNormGrad** is called to perform computation.

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

- **Parameters:**


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
      <th>Precaution</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>Main grad input. It corresponds to `inputdy` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>Input x1 of the forward fusion operator. It corresponds to `inputx1` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape and data type are the same as those of `dy`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input x2 of the forward fusion operator. It corresponds to `inputx2` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape and data type are the same as those of `dy`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Input</td>
      <td>Reciprocal of the standard deviation of the sum of forward inputs x1 and x2. It corresponds to `rstd` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must meet the <a href="../../../docs/en/context/broadcast_relationship.md">Broadcast Relationship</a> with `dy`. (The first several dimensions are the same as those of `dy`. The first several dimensions are the dimensions of `dy` minus the dimensions of `gamma`, indicating the dimensions that do not require normalization.)</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Mean value of the sum of forward inputs x1 and x2. It corresponds to `E(x)` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must meet the <a href="../../../docs/en/context/broadcast_relationship.md">Broadcast Relationship</a> with `dy`. (The first several dimensions are the same as those of `dy`. The first several dimensions are the dimensions of `dy` minus the dimensions of `gamma`, indicating the dimensions that do not require normalization.)</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>gamma of the forward input. It corresponds to `gamma` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type is the same as that of `dy`. </li><li>The dimension value of shape is the same as that of the dimension to be normalized in `dy`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dsumOptional</td>
      <td>Input</td>
      <td>Additional backward gradient accumulation input. It corresponds to `dsumOptional` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape and data type are the same as those of `dy`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dxOut</td>
      <td>Output</td>
      <td>Gradient of output `x` of the Add result. It corresponds to `dxOut` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape and data type are the same as those of `dy`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dgammaOut</td>
      <td>Output</td>
      <td>Gradient of the input parameter gamma. It corresponds to `dgammaOut` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape is the same as that of the input `gamma`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dbetaOut</td>
      <td>Output</td>
      <td>Backward gradient of the forward input parameter beta. It corresponds to `dbetaOut` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape is the same as that of the input `gamma`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
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
  </tbody>
  </table>

  - <term>Atlas inference series products</term>: The data types of `dy`, `x1`, `x2`, `gamma`, `dsumOptional`, and `dxOut` cannot be BFLOAT16.


- **Returns:**

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
      <td>The input parameter is a required input, output, or attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>The input or output data type is not supported.</td>
    </tr>
  </tbody></table>

## aclnnAddLayerNormGrad

- **Parameters:**

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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API aclnnAddLayerNormGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- **Function dimension**
  - The supported data types are as follows:
    - <term>Atlas inference series products</term>: dy, x1, x2, gamma, dsumOptional, and dxOut support FLOAT32 and FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: dy, x1, x2, gamma, dsumOptional, and dxOut support FLOAT32, FLOAT16, and BFLOAT16.
    - rstd, mean, dgammaOut, and dbetaOut support FLOAT32.
  - The data format can be ND.
- **Description of unsupported types**

  DOUBLE: The instructions do not support DOUBLE.

- **Description of boundary value scenarios**
  - When the input is Inf, the output is Inf.
  - When the input is NaN, the output is NaN.
- Deterministic compute:
  - **aclnnAddLayerNormGrad** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data from the host to the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
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
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output based on the API.
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

    // Create a self aclTensor.
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
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    // Call the first-phase API of aclnnAddLayerNormGrad.
    LOG_PRINT("\nUse aclnnAddLayerNormGrad Port.");
    ret = aclnnAddLayerNormGradGetWorkspaceSize(
        dy, x1, x2, rstd, mean, gamma, dsumOptional, outputpdx, outputpdgamma, outputpdbeta, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnAddLayerNormGrad.
    ret = aclnnAddLayerNormGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGrad failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
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

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
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

    // 7. Release device resources. Modify the configuration based on the API definition.
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
