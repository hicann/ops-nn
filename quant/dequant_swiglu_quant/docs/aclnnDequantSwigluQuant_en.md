# aclnnDequantSwigluQuant

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/quant/dequant_swiglu_quant)

## Product Support

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     ×    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     ×    |

## Function
- Description: Adds dequantization and quantization operations before and after the Swish-gated linear unit (SwiGLU) activation function, performing DequantSwiGLUQuant computation on input **x**. 
- Formula: 

  $$
  dequantOut = Dequant(x, weightScaleOptional, activationScaleOptional, biasOptional)
  $$

  $$
  swigluOut = Swiglu(dequantOut)=Swish(A)*B
  $$

  $$
  out = Quant(swigluOut, quantScaleOptional, quantOffsetOptional)
  $$

  Where **A** is the first half of **dequantOut** and B is the second half of **dequantOut**.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnDequantSwigluQuantGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnDequantSwigluQuant** is called to perform computation.

```Cpp
aclnnStatus aclnnDequantSwigluQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weightScaleOptional,
    const aclTensor *activationScaleOptional,
    const aclTensor *biasOptional,
    const aclTensor *quantScaleOptional,
    const aclTensor *quantOffsetOptional,
    const aclTensor *groupIndexOptional,
    bool             activateLeft,
    char            *quantModeOptional,
    const aclTensor *yOut,
    const aclTensor *scaleOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDequantSwigluQuant(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```


## aclnnDequantSwigluQuantGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
  <col style="width: 201px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 300px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
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
      <td>x</td>
      <td>Input</td>
      <td>Input data to be processed, corresponding to x in the formula.</td>
      <td>The shape is (N..., H). The last dimension must be a multiple of 2, and x must have at least 2 dimensions.</td>
      <td>FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
     <tr>
      <td>weightScaleOptional</td>
      <td>Input</td>
      <td>Dequantization scale for the weight, corresponding to weightScaleOptional in the formula.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
      <td>activationScaleOptional</td>
      <td>Input</td>
      <td>Dequantization scale for the activation function, corresponding to activationScaleOptional in the formula.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>biasOptional</td>
      <td>Input</td>
      <td>Bias of Matmul, corresponding to biasOptional in the formula.</td>
      <td>The shape can be 1D, denoted as [H], where H must match the last dimension of x. This parameter is optional and can be a null pointer.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>quantScaleOptional</td>
      <td>Input</td>
      <td>Quantization scale, corresponding to quantScaleOptional in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
       <tr>
      <td>quantOffsetOptional</td>
      <td>Input</td>
      <td>Quantization offset, corresponding to quantOffsetOptional in the formula.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
      <tr>
      <td>groupIndexOptional</td>
      <td>Input</td>
      <td>Group index required for MoE grouping.</td>
      <td>-</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>activateLeft</td>
      <td>Input</td>
      <td>Whether to apply SwiGLU activation to the left half of the input.</td>
      <td>When the value is false, activation is applied to the right half.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantModeOptional</td>
      <td>Input</td>
      <td>Whether to use dynamic or static quantization.</td>
      <td>"dynamic" and "static" are supported.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>yOut</td>
      <td>Output</td>
      <td>-</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOut</td>
      <td>Output</td>
      <td>-</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
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

    - weightScaleOptional:
      - <term>Atlas A2 training series products/Atlas A2 inference series products</term>: The data type can be FLOAT and the shape can be 1D, denoted as [H], where H must equal the last dimension of x. This parameter is optional and can be a null pointer.
    - activationScaleOptional:
      - <term>Atlas A2 training series products/Atlas A2 inference series products</term>: The data type can be FLOAT and the shape is [N..., 1], where the last dimension is 1 and the remaining dimensions are the same as x. This parameter is optional and can be a null pointer.
    - quantScaleOptional:
      - <term>Atlas A2 training series products/Atlas A2 inference series products</term>: The data type can be FLOAT or FLOAT16. When quantModeOptional is static, the shape is 1D with value 1, represented as shape[1]. When quantModeOptional is dynamic, the shape is 1D with value equal to half the last dimension of x, represented as shape[H/2]. This parameter is optional and can be a null pointer.
    - quantOffsetOptional:
      - <term>Atlas A2 training series products/Atlas A2 inference series products</term>: The data type can be FLOAT. When quantModeOptional is static, the shape is 1D with value 1, represented as shape[1]. When quantModeOptional is dynamic, the shape is 1D with value equal to half the last dimension of x, represented as shape[H/2]. This parameter is optional and can be a null pointer.
    - groupIndexOptional:
      - <term>Atlas A2 training series products/Atlas A2 inference series products</term>: The data type can be INT32 or INT64, and the shape can be a 1D tensor. This parameter is optional and can be a null pointer.
    - quantModeOptional:
      - <term>Atlas A2 training series products/Atlas A2 inference series products</term>: "dynamic" and "static" are supported.
    - yOut:
      - <term>Atlas A2 training series products/Atlas A2 inference series products</term>: The data type can be INT8.
    - scaleOut:
      - <term>Atlas A2 training series products/Atlas A2 inference series products</term>: The data type can be FLOAT.
- **Returns**

**aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
The first-phase API implements input parameter verification. The following errors may be thrown.
<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
<col style="width: 316px">
<col style="width: 111px">
<col style="width: 723px">
</colgroup>
<thead>
  <tr>
    <th>Return</th>
    <th>Error Code</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>The passed x, yOut, or scaleOut is a null pointer.</td>
  </tr>
  <tr>
    <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="4">161002</td>
    <td>The input or output data type is not supported.</td>
  </tr>
  <tr>
    <td>The input or output tensor rank is not supported.</td>
  </tr>
  <tr>
    <td>The input or output shape does not meet the constraints.</td>
  </tr>
  <tr>
    <td>The input value does not meet the requirements.</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_INNER_TILING_ERROR</td>
    <td>561002</td>
    <td>The memory size of the input tensor exceeds the upper limit.</td>
  </tr>
</tbody>
</table>

## aclnnDequantSwigluQuant

- **Parameters**
<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
<col style="width: 167px">
<col style="width: 123px">
<col style="width: 860px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnDequantSwigluQuantGetWorkspaceSize.</td>
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

- Deterministic computation:
  - **aclnnDequantSwigluQuant** defaults to a deterministic implementation.

- <term>Atlas A2 training series products/Atlas A2 inference series products</term>:
  - The last dimension of **x** must be a multiple of 2, and **x** must have at least two dimensions.
  - When **quantModeOptional** is **static**, **quantScaleOptional** and **quantOffsetOptional** are 1D with a value of 1. When **quantModeOptional** is **dynamic**, **quantScaleOptional** and **quantOffsetOptional** are 1D with a value equal to half the last dimension of **x**.
- <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The operator imposes an upper limit on the memory size of supported input tensors. The validation formula is: Memory size of weightScaleOptional + Memory size of biasOptional + Memory size of quantScaleOptional + Memory size of quantOffsetOptional + (Memory size of activationScaleOptional + Memory size of scaleOut)/40 + 10 × Memory size of the last dimension H of x < 192 KB.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```C++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dequant_swiglu_quant.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID (deviceId) based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on API definitions.
  std::vector<int64_t> xShape = {2, 32};
  std::vector<int64_t> scaleShape = {16};
  std::vector<int64_t> offsetShape = {1};
  std::vector<int64_t> outShape = {2, 16};
  std::vector<int64_t> scaleOutShape = {2};
  void* xDeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  void* offsetDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* scaleOutDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* scale = nullptr;
  aclTensor* offset = nullptr;
  aclTensor* out = nullptr;
  aclTensor* scaleOut = nullptr;
  std::vector<float> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  std::vector<float> scaleHostData = {1};
  std::vector<float> offsetHostData = {1};
  std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> scaleOutHostData = {0, 0};
  
  // Create an x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
   // Create a scale aclTensor.
  ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
   // Create an offset aclTensor.
  ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a scaleOut aclTensor.
  ret = CreateAclTensor(scaleOutHostData, scaleOutShape, &scaleOutDeviceAddr, aclDataType::ACL_FLOAT, &scaleOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API. Modify the API as required.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnDequantSwigluQuant.
  ret = aclnnDequantSwigluQuantGetWorkspaceSize(x, nullptr, nullptr, nullptr, scale, nullptr, nullptr, false, "dynamic", out, scaleOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSwigluQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnDequantSwigluQuant.
  ret = aclnnDequantSwigluQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSwigluQuant failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(scale);
  aclDestroyTensor(offset);
  aclDestroyTensor(out);
  aclDestroyTensor(scaleOut);
  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(xDeviceAddr);
  aclrtFree(scaleDeviceAddr);
  aclrtFree(offsetDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(scaleOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
