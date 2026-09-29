# aclnnInstanceNorm

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     ×    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     √    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Description: Performs instance normalization. Compared with [aclnnBatchNorm](../../batch_norm_v3/docs/aclnnBatchNorm_en.md), aclnnInstanceNorm normalizes each sample instance instead of the entire batch, making this function more suitable for processing data such as images.
- Formula:

  $$
  y = {{x-E(x)}\over\sqrt {Var(x)+eps}} * gamma + beta
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnInstanceNormGetWorkspaceSize** is called to obtain the input parameters and compute the workspace size required by the process. Then, **aclnnInstanceNorm** is called to perform computation.

```Cpp
aclnnStatus aclnnInstanceNormGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *gamma,
  const aclTensor *beta,
  const char      *dataFormat,
  double           eps,
  aclTensor       *y,
  aclTensor       *mean,
  aclTensor       *variance,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnInstanceNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnInstanceNormGetWorkspaceSize

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
      <td>Input for InstanceNorm computation, corresponding to x in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The actual data format is determined by dataFormat.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*) </td>
      <td>Input</td>
      <td>Scaling factor (weight) for InstanceNorm computation, corresponding to gamma in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of x. </li><li>The shape must be the same as the C axis of the input x.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta (aclTensor*) </td>
      <td>Input</td>
      <td>Bias for InstanceNorm computation, corresponding to beta in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of x. </li><li>The shape must be the same as the C axis of the input x.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dataFormat (char) </td>
      <td>Input</td>
      <td>Actual data format of the input tensor of the operator.</td>
      <td>NHWC or NCHW is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps (double) </td>
      <td>Input</td>
      <td>Value added to the variance to prevent division by zero, corresponding to eps in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Output result of InstanceNorm, corresponding to y in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape and data type must be the same as those of the input x.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean (aclTensor*) </td>
      <td>Output</td>
      <td>Mean value of InstanceNorm, corresponding to E(x) in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of the input x. </li><li>The shapes of mean and the input x must meet the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a>. (The shape of the first two dimensions must be the same as that of the first two dimensions of the input x. The first two dimensions represent the non-normalized dimensions, while the remaining dimensions must have a size of 1.)</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>variance (aclTensor*) </td>
      <td>Output</td>
      <td>Variance of InstanceNorm, corresponding to Var(x) in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of x. </li><li>The shapes of mean and the input x must meet the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a>. (The shape of the first two dimensions must be the same as that of the first two dimensions of the input x. The first two dimensions represent the non-normalized dimensions, while the remaining dimensions must have a size of 1.)</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>4</td>
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
      <td>The passed x, gamma, beta, y, mean, or variance is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of the passed x, gamma, beta, y, mean, or variance is not supported.</td>
    </tr>
    <tr>
      <td>x is not four-dimensional or gamma/beta is not one-dimensional.</td>
    </tr>
    <tr>
      <td>The shape of gamma/beta is inconsistent with the C axis of x.</td>
    </tr>
    <tr>
      <td>The product model is not supported.</td>
    </tr>
    <tr>
      <td>dataFormat is not set to NCHW or NHWC.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_NULLPTR</td>
      <td rowspan="2">561103</td>
      <td>The intermediate computation result of the aclnn API reports a null pointer.</td>
    </tr>
    <tr>
      <td>The size of the C axis or the H*W length for x and y is less than 32 bytes.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td>561101</td>
      <td>aclOpExecutor fails to be created in the API.</td>
    </tr>
  </tbody></table>

## aclnnInstanceNorm

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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnInstanceNormGetWorkspaceSize.</td>
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

  **aclnnStatus**: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Functional dimensions:
  - Supported data types:
    - x, gamma, beta, y, mean, and variance support FLOAT32 and FLOAT16.
  - The data format can be ND.
  - The shape of x and y must be four-dimensional. gamma/beta must be one-dimensional and consistent with the C axis of x and y.
  - The H\*W size of x and y must be greater than or equal to 32 bytes, and the C axis size must also be greater than or equal to 32 bytes.
  - dataFormat can only be NHWC or NCHW.
- Boundary value scenarios:
  - When the input is `Inf`, the output is `Inf`.
  - When the input is `NaN`, the output is `NaN`.
- Deterministic computation:
  - **aclnnInstanceNorm** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_instance_norm.h"

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

    int64_t N = 1;
    int64_t C = 8;
    int64_t H = 4;
    int64_t W = 4;

    // 2. Construct the input and output based on the API. In this example, the test cases with and without bias input are executed respectively.
    std::vector<int64_t> xShape = {N, C, H, W};
    std::vector<int64_t> weightShape = {C};
    std::vector<int64_t> yShape = {N, C, H, W};
    std::vector<int64_t> reduceShape = {N, C, 1, 1};

    void* xDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* varianceDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* y = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* variance = nullptr;

    std::vector<float> xHostData(N * C * H * W, 0.77);
    std::vector<float> gammaHostData(C, 1.5);
    std::vector<float> betaHostData(C, 0.5);
    std::vector<float> yHostData(N * C * H * W, 0.0);
    std::vector<float> meanHostData(N * C, 0.0);
    std::vector<float> varianceHostData(N * C, 0.0);
    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    // Create a self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, weightShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, weightShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(meanHostData, reduceShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(varianceHostData, reduceShape, &varianceDeviceAddr, aclDataType::ACL_FLOAT, &variance);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // aclnnInstanceNorm API call example
    // Call the first-phase API of aclnnInstanceNorm.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    LOG_PRINT("\nUse aclnnInstanceNorm Non-Bias Port.");
    ret = aclnnInstanceNormGetWorkspaceSize(
        x, gamma, beta, dataFormat, eps, y, mean, variance, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInstanceNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }

    // Call the second-phase API of aclnnInstanceNorm.
    ret = aclnnInstanceNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInstanceNorm failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.

    // 5.1 Copy the output without bias.
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== InstanceNorm non-bias: y output");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    auto outputMeanSize = GetShapeSize(reduceShape);
    std::vector<float> resultDataMean(outputMeanSize, 0);
    ret = aclrtMemcpy(
        resultDataMean.data(), resultDataMean.size() * sizeof(resultDataMean[0]), meanDeviceAddr,
        outputMeanSize * sizeof(resultDataMean[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== InstanceNorm non-bias: mean output");
    for (int64_t i = 0; i < outputMeanSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataMean[i]);
    }

    auto outputVarSize = GetShapeSize(reduceShape);
    std::vector<float> resultDataVar(outputVarSize, 0);
    ret = aclrtMemcpy(
        resultDataVar.data(), resultDataVar.size() * sizeof(resultDataVar[0]), varianceDeviceAddr,
        outputVarSize * sizeof(resultDataVar[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== InstanceNorm non-bias: rstd output");
    for (int64_t i = 0; i < outputVarSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataVar[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(y);
    aclDestroyTensor(mean);
    aclDestroyTensor(variance);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);

    aclrtFree(yDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(varianceDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
