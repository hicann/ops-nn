# aclnnQuantize

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/quant/quantize)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     √    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description: Quantizes the input tensor.
- Formula:
  
  $$
  out=round((x/scales)+zeroPoints)
  $$
  
## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnQuantizeGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnQuantize** is called to perform computation.

```Cpp
aclnnStatus aclnnQuantizeGetWorkspaceSize(
  const aclTensor* x,
  const aclTensor* scales,
  const aclTensor* zeroPoints,
  aclDataType      dtype,
  int32_t          axis,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnQuantize(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantizeGetWorkspaceSize

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
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Source data tensor to be quantized, corresponding to `x` in the formula.</td>
      <td>Empty tensors are supported.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Input</td>
      <td>scales tensor for x during quantization, corresponding to `scales` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The size must be 1 or the same as the size of axis in input x. </li><li>If the dtype of `x` is not FLOAT32, it must be the same as the dtype of `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints</td>
      <td>Input</td>
      <td>offset tensor for x during quantization, corresponding to `zeroPoints` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>Null pointers are supported. </li><li>The size must be 1 or the same as the size of axis in input x, and the same as the size of scales.</li></ul></td>
      <td>INT32, INT8, UINT8, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dstType</td>
      <td>Input</td>
      <td>Output data type.</td>
      <td>The data type can be ACL_INT8, ACL_UINT8, ACL_INT32, ACL_HIFLOAT8, ACL_FLOAT8_E4M3FN, or ACL_FLOAT8_E5M2.</td>
      <td>aclDataType</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Input</td>
      <td>element-wise axis to be quantized. Other axes are broadcast.</td>
      <td><ul><li>When the sizes of input scales and zeroPoints both are 1, this parameter is not used. </li><li>The value must be a negative number that is smaller than the number of dimensions of the input x and greater than or equal to the number of dimensions of x.</li></ul></td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Quantized output tensor, corresponding to `out` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as that of `x`, and the data type is specified by the input parameter `dstType`.</li></ul></td>
      <td>INT8, UINT8, INT32, HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
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
  
  - <term>Atlas inference series products</term>:
    - Data type:
      - The input parameters `x` and `scales` do not support BFLOAT16 or FLOAT32.
      - The input parameter `zeroPoints` does not support FLOAT32. When the data type is BFLOAT16, the data types of both `x` and `scales` are BFLOAT16.
      - The output parameter `out` supports only INT8, UINT8, and INT32.
    - The input parameter `dstType` supports only ACL_INT8, ACL_UINT8, and ACL_INT32.

  - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:
    - Data type:
      - The input parameter <idp:inline displayname="code" id="code184913122319">zeroPoints</idp:inline> does not support FLOAT32. When the data type is BFLOAT16, the data types of both `x` and `scales` are BFLOAT16.
      - The output parameter `out` supports only INT8, UINT8, and INT32.
    - The input parameter `dstType` supports only ACL_INT8, ACL_UINT8, and ACL_INT32.
  
- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown:

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
      <td>The passed x, scales, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type, data format, or dimension of x, scales, zeroPoints, or out is not supported.</td>
    </tr>
    <tr>
      <td>The axis specified by the input axis exceeds the number of dimensions of the input x.</td>
    </tr>
    <tr>
      <td>The input dtype is not supported.</tr>
    <tr>
      <td>The sizes of the input scales and zeroPoints are not equal.</td>
    </tr>
    <tr>
      <td>When the sizes of the input scales and zeroPoints are not 1, the sizes are not equal to the size of the axis specified by the input axis.</td>
    </tr>
    <tr>
      <td>The data types of out and dtype are different.</td>
    </tr>
  </tbody></table>

## aclnnQuantize

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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnQuantizeGetWorkspaceSize.</td>
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

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
  - **aclnnQuantize** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quantize.h"

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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output based on the API definition.
    std::vector<int64_t> xShape = {4, 2};
    std::vector<int64_t> scalesShape = {2};
    std::vector<int64_t> zeroPointsShape = {2};
    std::vector<int64_t> outShape = {4, 2};
    void* xDeviceAddr = nullptr;
    void* scalesDeviceAddr = nullptr;
    void* zeroPointsDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* scales = nullptr;
    aclTensor* zeroPoints = nullptr;
    aclTensor* out = nullptr;

    aclDataType dataType = ACL_INT32;
    int32_t axis = 1;
    std::vector<float> scalesHostData = {1.0, -3.0};
    std::vector<int32_t> zeroPointsData = {2, 10};
    std::vector<float> xHostData = {0.3382, -0.0919, 0.7564, 0.0234, 3.1024, 1.0761, 0.4228, 1.4621};
    std::vector<int32_t> outHostData = {8, 0};

    // Create an x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a scales aclTensor.
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a zeroPoints aclTensor.
    ret = CreateAclTensor(zeroPointsData, zeroPointsShape, &zeroPointsDeviceAddr, aclDataType::ACL_INT32, &zeroPoints);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, dataType, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnQuantize.
    ret = aclnnQuantizeGetWorkspaceSize(x, scales, zeroPoints, dataType, axis, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantizeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnQuantize.
    ret = aclnnQuantize(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantize failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
    auto outSize = GetShapeSize(outShape);
    std::vector<int32_t> outData(outSize, 0);
    ret = aclrtMemcpy(
        outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr, outSize * sizeof(outData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < outSize; i++) {
        LOG_PRINT("out[%ld] is: %d\n", i, outData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(scales);
    aclDestroyTensor(zeroPoints);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(scalesDeviceAddr);
    aclrtFree(zeroPointsDeviceAddr);
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
