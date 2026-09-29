# aclnnFastLayerNorm

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/layer_norm_v4)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Description: Performs normalization with a mean of 0 and a standard deviation of 1 on a specified layer. Compared with aclnnLayerNorm, aclnnFastLayerNorm improves the overall performance by 50%, and the memory usage is the same as that of the GPU. The accumulation order is optimized, resulting in different precision.

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

  E[x] indicates the input's mean value, and Var[x] indicates the input's variance.

## Prototype

Each operator has [two-phase API calls](../../../docs/en/context/two_phase_api.md). You must call the aclnnFastLayerNormGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the aclnnFastLayerNorm API to perform the computation.

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
      <td>input (aclTensor*) </td>
      <td>Input</td>
      <td>Input for normalization computation, corresponding to x in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape is [A1,...,Ai,R1,...,Rj], where A1 to Ai indicate dimensions that do not require normalization, and R1 to Rj indicate dimensions that require normalization.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>normalizedShape (aclIntArray*) </td>
      <td>Input</td>
      <td>Dimensions for normalization computation.</td>
      <td>The value is [R1,...,Rj]. The length is less than or equal to the length of the input shape. The value cannot be empty. In addition, R1*R2*...*Rj must be less than or equal to 583705600.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightOptional (aclTensor*) </td>
      <td>Input</td>
      <td>(Optional) Weight for normalization computation. It corresponds to weightOptional in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>When weightOptional is not empty: <ul><li>The data type is the same as that of input or is FLOAT. When biasOptional exists, the data types of weightOptional and biasOptional are the same. </li><li>The shape is the same as that of normalizedShape, that is, [R1,...,Rj]. </li></ul></li><li>When weightOptional is empty, the API constructs a tensor with shape [R1,...,Rj] and all data being 1. <ul><li>When biasOptional exists, the data types of weightOptional and biasOptional are the same. </li><li>When biasOptional does not exist, the data types of weightOptional and input are the same.</li></ul></li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensor*) </td>
      <td>Input</td>
      <td>(Optional) Offset for normalization computation. It corresponds to biasOptional in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>When biasOptional is not empty: <ul><li>The data type is the same as that of input or is FLOAT. When weightOptional exists, the data types of biasOptional and weightOptional are the same. </li><li>The shape is the same as that of normalizedShape, that is, [R1,...,Rj]. </li></ul></li><li>When biasOptional is empty, the API constructs a tensor with shape [R1,...,Rj] and all data being 0. <ul><li>When weightOptional exists, the data types of biasOptional and weightOptional are the same. </li><li>When weightOptional does not exist, the data types of biasOptional and input are the same.</li></ul></li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>eps (double) </td>
      <td>Input</td>
      <td>Value added to the denominator to ensure numerical stability. corresponding to eps in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*) </td>
      <td>Output</td>
      <td>Normalization result. It corresponds to out in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of input. </li><li>The shape must be the same as that of input, that is, [A1,...,Ai,R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>meanOutOptional (aclTensor*) </td>
      <td>Output</td>
      <td>(Optional) Normalized mean value. It corresponds to `meanOutOptional` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of input. </li><li>When rstdOutOptional exists, the shape is the same as that of rstdOutOptional. The shape is [A1,...,Ai,1,...,1] with j 1s after Ai, which is the same as the length of the axis to be normalized.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOutOptional (aclTensor*) </td>
      <td>Output</td>
      <td>(Optional) Reciprocal of the normalized standard deviation. It corresponds to `rstdOutOptional` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of input. </li><li>When meanOutOptional exists, the shape is the same as that of meanOutOptional. The shape is [A1,...,Ai,1,...,1] with j 1s after Ai, which is the same as the length of the axis to be normalized.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*) </td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**) </td>
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

- **Return Value**

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
      <td>The passed input, normalizedShape, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The number of dimensions of the input, weightOptional (when it is not null), biasOptional (when it is not null), out, meanOutOptional (when it is not null), rstdOutOptional (when it is not null), and shape exceeds 8.</td>
    </tr>
    <tr>
      <td>The number of elements in normalizedShape exceeds 8.</td>
    </tr>
    <tr>
      <td>The data type of input, weightOptional (non-empty), biasOptional (non-empty), out, meanOutOptional (non-empty), or rstdOutOptional (non-empty) is not supported.</td>
    </tr>
    <tr>
      <td>normalizedShape has less than one dimension.</td>
    </tr>
    <tr>
      <td>weightOptional is specified and the shape is different from normalizedShape.</td>
    </tr>
    <tr>
      <td>biasOptional is specified and the shape is different from normalizedShape.</td>
    </tr>
    <tr>
      <td>The number of dimensions of input is less than that of normalizedShape.</td>
    </tr>
    <tr>
      <td>The shape of input is different from the shape of the corresponding dimension when right aligned with normalizedShape.</td>
    </tr>
    <tr>
      <td>The shapes of `input` and `out` are inconsistent.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>When the value of normalizedShape is [R1,..., Rj], the value of R1*R2*...*Rj is greater than 583705600.</td>
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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnFastLayerNormGetWorkspaceSize.</td>
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

- **Return Value**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- The shape of input, normalizedShape, weightOptional (non-empty), biasOptional (non-empty), out, meanOutOptional (non-empty), or rstdOutOptional (non-empty) cannot exceed eight dimensions.
- Deterministic computation:
  - The default deterministic implementation of aclnnFastLayerNorm is used.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
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

    // 3. Call the CANN operator library API. Modify the API name to the actual one.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first part of the aclnnFastLayerNorm API.
    ret = aclnnFastLayerNormGetWorkspaceSize(x, norm, weight, bias, eps, out, mean, rstd, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFastLayerNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second part of the aclnnFastLayerNorm API.
    ret = aclnnFastLayerNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFastLayerNorm failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
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

    // 6. Release the aclTensor and aclIntArray. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyIntArray(norm);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(out);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);

    // 7. Free device resources.
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
