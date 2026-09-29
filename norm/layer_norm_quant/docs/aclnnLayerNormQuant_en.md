# aclnnLayerNormQuant

[📄 View Source Code](https://gitcode.com/cann/ops-nn/tree/master/norm/layer_norm_quant)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- API function: The LayerNorm operator is a common normalization operation used in large models. The LayerNormQuant operator integrates the output of the LayerNorm normalization with the downstream quantization operator to reduce the data transfer operations.
- Formula:
  * LayerNorm operation:
  
    $$
    y = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * gamma + beta
    $$
    
    $$
    E(x) = {\frac{1}{n} \sum_{i=1}^{n} x_i }
    $$
    
    $$
    Var(x) = {\frac{1}{n} \sum_{i=1}^{n} (x_i-E(x))^2 }
    $$
  
  * When quantMode is set to 0, the quantization mode is static quantization, and the output scaleOut is meaningless.
    
    $$
    res = y / scale + zeroPointsOptional
    $$

  * When quantMode is set to 1, the quantization mode is dynamic quantization.
  
    $$
    tmp = y * scale
    $$
    
    $$
    scaleOut = row\_max(abs(tmp))/dtypeMax
    $$
    
    $$
    res = round(y / scaleOut )
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnLayerNormQuantGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnLayerNormQuant` is called to perform computation.

```Cpp
aclnnStatus aclnnLayerNormQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *gamma,
  const aclTensor *beta,
  const aclTensor *scale,
  const aclTensor *zeroPointsOptional,
  int              quantMode,
  double           epsilon,
  aclTensor       *res,
  aclTensor       *scaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnLayerNormQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnLayerNormQuantGetWorkspaceSize

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
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td> indicates the x parameter in layer normalization. It corresponds to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*) </td>
      <td>Input</td>
      <td>gamma parameter for layer normalization. It corresponds to `gamma` in the formula.</td>
      <td><ul><li>Empty tensors are supported. When </li><li>quantMode is 0, the shape supports two dimensions and the first dimension is 1. When </li><li>quantMode is 1, the shape must be the same as the x dimension. Except the last dimension, other dimensions are 1. </li><li>The last dimension must be the same as that of `x`. </li><li>The data type must be the same as that of x.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta (aclTensor*) </td>
      <td>Input</td>
      <td>Beta parameter in the LayerNorm formula, indicating the beta parameter in layer normalization. It corresponds to `beta` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape is the same as that of `gamma`. </li><li>The data type must be the same as that of x.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale (aclTensor*) </td>
      <td>Input</td>
      <td>Scale input in the fused quantized computation. It corresponds to scale in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape is [1] and the dimension is 1.</li><li>The data type must be the same as that of x.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPointsOptional (aclTensor*) </td>
      <td>Input</td>
      <td>Optional zloss-related input. Indicates the zeroPointsOptional input in the fused quantized computation. This parameter is valid only when quantMode is set to 0, corresponding to `zeroPointsOptional` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as that of scale.</li></ul></td>
      <td>INT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantMode (int) </td>
      <td>Input</td>
      <td>Quantization mode, which is used to determine whether the fusion operator is static or dynamic. It corresponds to `quantMode` in the formula. The value can be 0 (static quantization) or 1 (dynamic quantization).</td>
      <td>Currently, only the value 0 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon (double) </td>
      <td>Input</td>
      <td>Epsilon in LayerNorm, which is added to the denominator to ensure numerical stability. corresponding to `epsilon` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>res (aclTensor*) </td>
      <td>Output</td>
      <td>Quantized result of the LayerNorm output y. It corresponds to `res` in the formula.</td>
      <td>The shape must be the same as that of the input x.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOut (aclTensor*) </td>
      <td>Output</td>
      <td>ScaleOut result of dynamic quantization, corresponding to `scaleOut` in the formula. This parameter is valid only when quantMode is set to 1.</td>
      <td>The shape is the shape of x with the last dimension removed.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-7</td>
      <td>√</td>
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
  </tbody>
  </table>
  
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
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
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The hardware platform is not supported.</td>
    </tr>
    <tr>
      <td>The input data type does not meet the constraints.</td>
    </tr>
    <tr>
      <td>The last dimension of gamma is inconsistent with that of x.</td>
    </tr>
    <tr>
      <td>The shapes of x and res are different.</td>
    </tr>
    <tr>
      <td>The shapes of gamma and beta are different.</td>
    </tr>
    <tr>
      <td>The shapes of zeroPointsOptional and scale are different.</td>
    </tr>
    <tr>
      <td>The value of quantMode is not 0.</td>
    </tr>
  </tbody></table>

## aclnnLayerNormQuant

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
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnLayerNormQuantGetWorkspaceSize.</td>
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

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - The aclnnLayerNormQuant is implemented in deterministic mode by default.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_layer_norm_quant.h"

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
    // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
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

    // 2. Construct the inputs and outputs based on the API definition.
    float eps = 1e-6;
    int quantMode = 0;

    std::vector<int64_t> xShape = {1, 1, 32};
    std::vector<int64_t> gammaShape = {1, 32};
    std::vector<int64_t> betaShape = {1, 32};
    std::vector<int64_t> scaleOptionalShape = {1};
    std::vector<int64_t> zeroPointOptionalShape = {1};

    std::vector<int64_t> outputYShape = {1, 1, 32};
    std::vector<int64_t> outputScaleShape = {1, 1};

    void* xDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* scaleOptionalDeviceAddr = nullptr;
    void* zeroPointOptionalDeviceAddr = nullptr;

    void* outputYDeviceAddr = nullptr;
    void* outputScaleDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* scaleOptional = nullptr;
    aclTensor* zeroPointOptional = nullptr;

    aclTensor* outputY = nullptr;
    aclTensor* outputScale = nullptr;

    std::vector<float> xHostData = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> gammaHostData = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> betaHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> scaleOptionalHostData = {1};
    std::vector<int8_t> zeroPointOptionalHostData = {1};

    std::vector<int8_t> outputYHostData(1 * 1 * 32);
    std::vector<float> outputScaleHostData(1);

    // Create a self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleOptionalHostData, scaleOptionalShape, &scaleOptionalDeviceAddr, aclDataType::ACL_FLOAT, &scaleOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        zeroPointOptionalHostData, zeroPointOptionalShape, &zeroPointOptionalDeviceAddr, aclDataType::ACL_INT8, &zeroPointOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outputYHostData, outputYShape, &outputYDeviceAddr, aclDataType::ACL_INT8, &outputY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, aclDataType::ACL_FLOAT, &outputScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Example of calling the aclnnLayerNormQuant API
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.

    // Call the first part of the aclnnLayerNormQuant API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    LOG_PRINT("\nUse aclnnLayerNormQuant Port.");
    ret = aclnnLayerNormQuantGetWorkspaceSize(
        x, gamma, beta, scaleOptional, zeroPointOptional, quantMode, eps, outputY, outputScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second API call of aclnnLayerNormQuant.
    ret = aclnnLayerNormQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormQuant failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto outputYSize = GetShapeSize(outputYShape);
    std::vector<int8_t> resultDataY(outputYSize, 0);
    ret = aclrtMemcpy(
        resultDataY.data(), resultDataY.size() * sizeof(resultDataY[0]), outputYDeviceAddr,
        outputYSize * sizeof(resultDataY[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < outputYSize; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultDataY[i]);
    }

    if (quantMode == 1){
        auto outputScaleSize = GetShapeSize(outputScaleShape);
        std::vector<float> resultDataScale(outputScaleSize, 0);
        ret = aclrtMemcpy(
            resultDataScale.data(), resultDataScale.size() * sizeof(resultDataScale[0]), outputScaleDeviceAddr,
            outputScaleSize * sizeof(resultDataScale[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < outputScaleSize; i++) {
            LOG_PRINT("result[%ld] is: %f\n", i, resultDataScale[i]);
        }
    }

    // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(scaleOptional);
    aclDestroyTensor(zeroPointOptional);

    aclDestroyTensor(outputY);
    aclDestroyTensor(outputScale);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(scaleOptionalDeviceAddr);
    aclrtFree(zeroPointOptionalDeviceAddr);

    aclrtFree(outputYDeviceAddr);
    aclrtFree(outputScaleDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
