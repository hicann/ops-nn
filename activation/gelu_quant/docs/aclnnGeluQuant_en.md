# aclnnGeluQuant

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     √    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     ×    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- This API is used to fuse GeluV2 with DynamicQuant/AscendQuantV2. It performs Gelu activation on the input data self, quantizes the activation result, and outputs the quantized result.

- Formula:

1. Calculate the Gelu value to obtain geluOut.

    - approximate = tanh

    $$
    geluOut=Gelu(self)=self × Φ(self)=0.5 * self * (1 + Tanh( \sqrt{2 / \pi} * (self + 0.044715 * self^{3})))
    $$

    - approximate = none

    $$
    geluOut=Gelu(self)=self × Φ(self)=0.5 * self *[1 + erf(self/\sqrt{2})]
    $$

2. Quantize geluOut.

    - quant_mode = static

    $$
    y = round\_to\_dst\_type(geluOut * inputScaleOptional + inputOffsetOptional, round\_mode)
    $$

    - quant_mode = dynamic

      $$
      geluOut = geluOut * inputScaleOptional
      $$

      $$
      Max = max(abs(geluOut))
      $$

      $$
      outScaleOptional = Max/maxValue
      $$
      
      $$
      y = round\_to\_dst\_type(geluOut / outScaleOptional, round\_mode)
      $$
    
    - maxValue: maximum value of the corresponding data type.
    
      |   DataType    | maxValue |
      | :-----------: | :------: |
      |     INT8      |  127    |
      | FLOAT8_E4M3FN |  448   |
      |  FLOAT8_E5M2  |  57344  |
      |   HIFLOAT8    |  32768   |

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. You must call aclnnGeluQuantGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call aclnnGeluQuant to perform the computation.

```Cpp
aclnnStatus aclnnGeluQuantGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* inputScaleOptional,
    const aclTensor* inputOffsetOptional,
    const char*      approximate,
    const char*      quantMode,
    const char*      roundMode,
    int64_t          dstType,
    const aclTensor* y,
    const aclTensor* outScaleOptional,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnGeluQuant(
    void*            workspace,
    uint64_t         workspaceSize,
    aclOpExecutor*   executor,
    aclrtStream      stream)
```

## aclnnGeluQuantGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 301px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 320px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
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
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Input `self` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>When quantMode is set to "dynamic", the shape supports 2 to 8 dimensions. </li><li>When quantMode is set to "static", the shape supports 1 to 8 dimensions.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
     <tr>
      <td>inputScaleOptional (aclTensor*) </td>
      <td>Input</td>
      <td>Input of the operator, which is inputScaleOptional in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape supports only one dimension. The size can only be the size of the last axis of the self or 1. </li><li>This input is required when quantMode is set to static, and is optional when quantMode is set to dynamic.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
     <tr>
      <td>inputOffsetOptional (aclTensor*) </td>
      <td>Input</td>
      <td>Optional input of the operator, which is the inputOffsetOptional in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape supports only one dimension, and must be the same as the dtype and shape of inputScaleOptional. </li><li>When quantMode is set to dynamic and inputScaleOptional is not input, offset is not input.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>approximate (char*)</td>
      <td>Input</td>
      <td>approximate in the formula, which is the mode of the gelu activation function.</td>
      <td>approximate supports only {"none", "tanh"}.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>quantMode (char*) </td>
      <td>Input</td>
      <td>Quantization mode in the formula.</td>
      <td>The quantization mode can be either static or dynamic, which corresponds to {"static", "dynamic"}.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>roundMode (char*) </td>
      <td>Input</td>
      <td>Gelu activation function mode in the formula.</td>
      <td><ul><li>{"rint", "round", "hybrid"} are supported. </li><li>If dstType is 2/35/36 and the corresponding data type is INT8/FLOAT8_E4M3FN/FLOAT8_E5M2, only {"rint"} is supported. </li><li>If dstType is 34 and the corresponding data type is HIFLOAT8, {"round", "hybrid"} is supported.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>dstType (int64_t) </td>
      <td>Input</td>
      <td>dst_type in the formula.</td>
      <td>: Specifies the type of y after data conversion. The input range is {2, 34, 35, 36}, corresponding to the data type {2: INT8, 34: HIFLOAT8, 35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN} of the output y.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>After the activation, the quantized result is output, that is, y in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li> The data type must be the same as that of dstType. </li><li> The value must be the same as the shape size of self.</li></ul></td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>outScaleOptional (aclTensor*) </td>
      <td>Output</td>
      <td>Quantization scale of dynamic quantization, that is, outScaleOptional in the formula.</td>
      <td><ul><li>Empty tensors are not supported. The dimension of </li><li>shape is one less than that of self. </li><li> The dimension size is the same as that of self except the last dimension. </li><li> When quantMode is set to static, the output of outScaleOptional should be a null pointer.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-7</td>
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

`aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

The first-phase API implements input parameter validation. The following error codes may be returned.

<table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
<col style="width: 319px">
<col style="width: 108px">
<col style="width: 621px">
</colgroup>
<thead>
  <tr>
    <th>Return Code</th>
    <th>Error Code</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td><ul><li>The input self is a null pointer. </li><li>When quantMode is set to static, inputScaleOptional and inputOffsetOptional are null pointers.</li></ul></td>
  </tr>
  <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
      <td><ul><li>self, inputScaleOptional, inputOffsetOptional, y, and outScaleOptional are null tensors. </li><li>The data types of self, inputScaleOptional, inputOffsetOptional, y, and outScaleOptional are not supported. </li><li>approximate, quantMode, roundMode, and dstType are not supported. </li><li>The shapes of self, inputScaleOptional, inputOffsetOptional, y, and outScaleOptional do not meet the verification conditions.</li></ul></td>
  </tr>
  <tr>
    <td rowspan="3">ACLNN_ERR_RUNTIME_ERROR</td>
    <td rowspan="3">361001</td>
    <td>The current platform is not supported.</td>
  </tr>
</tbody>
</table>

## aclnnGeluQuant

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
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnGeluQuantGetWorkspaceSize API.</td>
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

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - The aclnnGeluQuant is implemented in deterministic mode by default.

- The data type of inputScaleOptional is the same as that of self. If the types are different, the type with higher precision is used.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).
  
```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_gelu_quant.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
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
    // Fixed format, resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnGeluQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> xShape = {4, 2};
    std::vector<int64_t> inputScaleShape = {1};
    std::vector<int64_t> inputOffsetShape = {1};
    std::vector<int64_t> yOutShape = {4, 2};
    std::vector<int64_t> emptyShape = {4, 0};
    void* xDeviceAddr = nullptr;
    void* inputScaleDeviceAddr = nullptr;
    void* inputOffsetDeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    void* outscaleOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* inputScale = nullptr;
    aclTensor* inputOffset = nullptr;
    aclTensor* yOut = nullptr;
    aclTensor* outScale = nullptr;
    std::vector<float> xHostData = {1.3, 2.5, 6.7, -4, -1.4, -1.6, -8, -16.9};
    std::vector<float> inputScaleHostData = {1};
    std::vector<uint8_t> yOutHostData = {1, 2, 7, 0, 0, 0, 0, 0};
    const char* approximate = "tanh";
    const char* quantMode = "static";
    const char* roundMode = "rint";
    int64_t dstType = 2;
    // Create an x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an inputScale aclTensor.
    ret = CreateAclTensor(inputScaleHostData, inputScaleShape, &inputScaleDeviceAddr, aclDataType::ACL_FLOAT, &inputScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> inputScaleTensorPtr(inputScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> inputScaleDeviceAddrPtr(inputScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create yOut aclTensor.
    ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_INT8, &yOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    // Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first part of the aclnnGeluQuant API.
    ret = aclnnGeluQuantGetWorkspaceSize(x, inputScale, inputOffset, approximate, quantMode, roundMode, dstType, yOut, outScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGeluQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second segment of the aclnnGeluQuant API.
    ret = aclnnGeluQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGeluQuant failed. ERROR: %d\n", ret); return ret);

    // (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(yOutShape);
    std::vector<int8_t> yOutData(
        size, 0); 
    ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                    size * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
            return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("y[%ld] is: %d\n", i, yOutData[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGeluQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGeluQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
