# aclnnQuantizedBatchNorm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:26:49.150Z -->

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

- Description:

  Performs dequantization on the input Tensor, then normalization based on the input weight, bias, and epsilon, and finally quantization based on the input outputScale and outputZeroPoint.
- Formula:

  1. Dequantization computation:

     $$
     x' = (x - inputZeroPoint) * inputScale
     $$

  2. Normalization calculation:

     $$
     y =\frac{x' - mean}{\sqrt{var + epsilon}} * weight + bias
     $$

  3. Quantization calculation:

     $$
     output = round(\frac{y}{outputScale} + outputZeroPoint)
     $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnQuantizedBatchNormGetWorkspaceSize" API to obtain input parameters and calculate the required workspace size based on the computation flow, and then call the "aclnnQuantizedBatchNorm" API to perform the computation.

```Cpp
aclnnStatus aclnnQuantizedBatchNormGetWorkspaceSize(
  const aclTensor* input,
  const aclTensor* mean,
  const aclTensor* var,
  const aclScalar* inputScale,
  const aclScalar* inputZeroPoint,
  const aclScalar* outputScale,
  const aclScalar* outputZeroPoint,
  aclTensor*       weight,
  aclTensor*       bias,
  float            epsilon,
  aclTensor*       output,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnQuantizedBatchNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantizedBatchNormGetWorkspaceSize

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
      <td>Quantized data of the model input, corresponding to `x` in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT8, UINT8, INT32</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean (aclTensor*)</td>
      <td>Input</td>
      <td>Mean of the model input data, corresponding to `mean` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape length is equal to the channel axis length of the input parameter `input`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>var (aclTensor*)</td>
      <td>Input</td>
      <td>Variance of the model input data, corresponding to `var` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape length is equal to the channel axis length of the input parameter `input`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>inputScale (aclScalar*)</td>
      <td>Input</td>
      <td>Scale factor of the model input data, corresponding to `inputScale` in the formula.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputZeroPoint (aclScalar*)</td>
      <td>Input</td>
      <td>Represents the offset of the model input data, corresponding to `inputZeroPoint` in the formula.</td>
      <td>The value passed in cannot exceed the upper and lower bounds of the data type of `input`. For example, the upper and lower bounds of INT8 are [-128,127].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputScale (aclScalar*)</td>
      <td>Input</td>
      <td>Represents the scaling factor of the model output data, corresponding to `outputScale` in the formula.</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputZeroPoint (aclScalar*)</td>
      <td>Input</td>
      <td>Represents the offset of the model output data, corresponding to `outputZeroPoint` in the formula.</td>
      <td>The value cannot exceed the upper and lower bounds of the data type of `input`. For example, the bounds of INT8 are [-128,127].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight (aclTensor*)</td>
      <td>Input</td>
      <td>Optional parameter. Indicates the normalization weight, corresponding to `weight` in the formula.</td>
      <td><ul><li>Empty Tensor not supported.</li><li>Default value: 1.</li><li>The shape length is equal to the channel axis length of input parameter `input`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>bias (aclTensor*)</td>
      <td>Input</td>
      <td>Optional parameter, indicating the normalization bias, corresponding to `bias` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Default value: 0.</li><li>The shape length is equal to the channel axis length of the input parameter `input`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon (float)</td>
      <td>Input</td>
      <td>Represents the value added to the variance to avoid division by zero. Corresponds to `epsilon` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantized data of the model output, corresponding to `output` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape, data format, and data type are the same as those of input `input`.</li></ul></td>
      <td>INT8, UINT8, INT32</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied for by the user on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor that contains the operator computation flow.</td>
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
      <td>The input or output is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type is not in the supported range, or the parameters do not meet the preceding constraints.</td>
    </tr>
  </tbody></table>

## aclnnQuantizedBatchNorm

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
      <td>Device side-applied workspace size, obtained by the first-phase API aclnnQuantizedBatchNormGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>An op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>The Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnQuantizedBatchNorm defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quantized_batch_norm.h"

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
    // Boilerplate: initialize resources.
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

    // Call aclrtMemcpy to copy the data on the host to the device memory.
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

template <typename T>
int CreateAclTensorNCHW(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy the data on the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
    // Set deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Customize the construction based on the API.
    std::vector<int64_t> selfShape = {1, 2, 1, 4};
    std::vector<int64_t> meanShape = {2};
    std::vector<int64_t> varShape = {2};
    std::vector<int64_t> weightShape = {2};
    std::vector<int64_t> biasShape = {2};
    std::vector<int64_t> outShape = {1, 2, 1, 4};
    void* selfDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* rMeanDeviceAddr = nullptr;
    void* rVarDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* varDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* out = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* var = nullptr;
    std::vector<int32_t> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> weightHostData = {1, 1};
    std::vector<float> biasHostData = {0, 0};
    std::vector<float> meanHostData = {0, 0};
    std::vector<float> varHostData = {1, 1};
    std::vector<int32_t> outHostData(8, 0);
    float inputScaleValue = 1.0f;
    int32_t inputZeroPointValue = 1;
    float outputScaleValue = 2.0f;
    int32_t outputZeroPointValue = 1;
    double eps = 1e-5;

    // Create the self aclTensor.
    ret = CreateAclTensorNCHW(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the mean aclTensor.
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the var aclTensor.
    ret = CreateAclTensor(varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the bias aclTensor.
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensorNCHW(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the inputScale aclScalar.
    aclScalar* inputScale = aclCreateScalar(&inputScaleValue, aclDataType::ACL_FLOAT);
    // Create the inputZeroPoint aclScalar.
    aclScalar* inputZeroPoint = aclCreateScalar(&inputZeroPointValue, aclDataType::ACL_INT32);
    // Create the outputScale aclScalar.
    aclScalar* outputScale = aclCreateScalar(&outputScaleValue, aclDataType::ACL_FLOAT);
    // Create the outputZeroPoint aclScalar.
    aclScalar* outputZeroPoint = aclCreateScalar(&outputZeroPointValue, aclDataType::ACL_INT32);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Example of calling the aclnnQuantizedBatchNorm API
    // 3. Call the CANN operator library API. Change the API name to the actual one.
    // Call the first-phase API of aclnnQuantizedBatchNorm.
    ret = aclnnQuantizedBatchNormGetWorkspaceSize(
        self, mean, var, inputScale, inputZeroPoint, outputScale, outputZeroPoint, weight, bias, eps, out,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantizedBatchNormGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnQuantizedBatchNorm.
    ret = aclnnQuantizedBatchNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantizedBatchNorm failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Wait synchronously for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from Device memory to the Host side. Change the code based on the actual API definition.
    auto size = GetShapeSize(outShape);
    printf("size is %ld", size);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify based on the specific API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(mean);
    aclDestroyTensor(var);
    aclDestroyTensor(out);
    aclDestroyScalar(inputScale);
    aclDestroyScalar(inputZeroPoint);
    aclDestroyScalar(outputScale);
    aclDestroyScalar(outputZeroPoint);

    // 7. Release device resources. Modify based on the specific API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(varDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
