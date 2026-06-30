# aclnnAscendQuantV3

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/quant/ascend_quant_v2)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     √    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description: Quantizes the input x. The axes of scale and offset can be set by using axis. The shapes of scale and offset must be the same as axes of x specified by axis or equal to 1. Currently, the last two dimensions of axis can be set.
- Formula:
  - When **sqrtMode** is **false**, the calculation formula is as follows:

    $$
    y = round((x * scale) + offset)
    $$

  - When **sqrtMode** is **true**, the calculation formula is as follows:

    $$
    y = round((x * scale * scale) + offset)
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnAscendQuantV3GetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnAscendQuantV3** is called to perform computation.

```Cpp
aclnnStatus aclnnAscendQuantV3GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *scale,
  const aclTensor *offset,
  bool             sqrtMode,
  const char      *roundMode,
  int32_t          dstType,
  int32_t          axis,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAscendQuantV3(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAscendQuantV3GetWorkspaceSize

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
      <td>x</td>
      <td>Input</td>
      <td>Input to be quantized. It corresponds to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>When the data format is ND, the last dimension of the shape must be divisible by 8 if `dstType` is 3 and be divisible by 2 if `dstType` is 29. </li><li>When the data format is NZ, the shape can only be 3D, and the last dimension of the shape must be divisible by 8.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NZ</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Scale value for quantization. It corresponds to `scale` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>`scale` supports 1D or multi-dimensional tensors. The shape is related to the input `x` and attribute `axis`. (When the shape of `scale` is a 1D tensor, the 0th dimension of `scale` must be equal to the `axis`th dimension of x or equal to 1. When the shape of `scale` is a multi-dimensional tensor, the number of dimensions of `scale` must be the same as that of `x`. The `axis`th dimension of `scale` must be equal to the `axis`th dimension of x or equal to 1, and other dimensions of `scale` must be 1.) </li><li>The data type and format must be the same as those of `x`. </li><li>When the data format is NZ, all element values of `scale` are 1.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NZ</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td>(Optional) Offset value for dequantization. It corresponds to `offset` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type and shape must be the same as those of `scale`. </li><li>When the data format is NZ, the value is empty, and the data type of offset is the same as that of x.</li></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NZ</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sqrtMode</td>
      <td>Input</td>
      <td>Compute logic of scale. It corresponds to `sqrtMode` in the formula.</td>
      <td>If the value is true, the formula is y = round((x × scale × scale) + offset). If the value is false, the formula is y = round((x × scale) + offset).</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundMode</td>
      <td>Input</td>
      <td>Mode of casting or converting into INT8 output.</td>
      <td><ul><li>The value can be round, ceil, trunc, floor, or hybrid. </li><li>When the data format of input `x` is NZ, the value can be round.</li></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType</td>
      <td>Input</td>
      <td>Output data type.</td>
      <td><ul><li>The value can be 2, 3, 29, 34, 35, or 36, indicating INT8, INT32, INT4, HIFLOAT8, FLOAT8_E5M2, or FLOAT8_E4M3FN, respectively. </li><li>When the data format of input `x` is NZ, the value can be 3, indicating INT32.</li></td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>  
    <tr>
      <td>axis</td>
      <td>Input</td>
      <td>Dimensions of `scale` and `offset` corresponding to those of `x`. When the data format of the input `x` is NZ, the value is -1.</td>
      <td>-</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr> 
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Compute output for quantization. It corresponds to `y` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>If the type is INT32, the last dimension of the shape is 1/8 of the last dimension of `x`, and other dimensions are the same as those of `x`. For other types, the shape is the same as that of `x`.</li></td>
      <td>INT8, INT32, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND, NZ</td>
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

  - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:

    - When the data format of `x`, `scale`, and `offset` is NZ, the data type can only be FLOAT32.
    - The data type of the output parameter `y` can only be INT8, INT32, or INT4. When the data format is NZ, the data type can be INT32.
    - The input parameter `roundMode` can be set to round, ceil, trunc, or floor. When the data format of the input `x` is NZ, the value can be round.
    - The input parameter `dstType` can be set to 2, 3, or 29, indicating INT8, INT32, or INT4, respectively. When the data format of input `x` is NZ, the value can be 3, indicating INT32.
    - The input parameter `axis` can only specify the last two dimensions of x. (Assume that the dimension of input x is xDimNum. The value range of axis is [–2, –1] or [xDimNum–2, xDimNum–1].)


  - <term>Atlas inference series products</term>:
    - The data types of the input parameters `x`, `scale`, and `offset` do not support BFLOAT16, and the data format cannot be NZ.
    - The data type of the output parameter `y` can only be INT8, and the data format cannot be NZ.
    - The input parameter `roundMode` can be set to round, ceil, trunc, or floor.
    - The input parameter `dstType` can only be set to 2, indicating INT8.
    - The input parameter `axis` can only specify the last dimension of x. (Assume that the dimension of input x is xDimNum. The value of axis is –1 or xDimNum–1.)


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
      <td>The passed x, scale, or y is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type or format of x, scale, offset, or y is not supported.</td>
    </tr>
    <tr>
      <td>The shape of x, scale, offset, or y does not meet the constraints.</td>
    </tr>
    <tr>
      <td>The number of dimensions of x is not in the range from 1 to 8.</td>
    </tr>
    <tr>
      <td>The value of roundMode is invalid.</tr>
    <tr>
      <td>The value of dstType is invalid.</td>
    </tr>
    <tr>
      <td>The value of axis is invalid.</td>
    </tr>
    <tr>
      <td>If the data type of y is INT4, the size of the last axis of x is not an even number.</td>
    </tr>
    <tr>
      <td>If the data type of y is INT32, the size of the last axis of the shape of y is not 1/8 of that of x, or the sizes of the non-last axes of the shapes of x and y are inconsistent.</td>
    </tr>
  </tbody></table>

## aclnnAscendQuantV3

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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API aclnnAscendQuantV3GetWorkspaceSize.</td>
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

- Deterministic compute:
  - **aclnnAscendQuantV3** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ascend_quant_v3.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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

    // Call aclrtMemcpy to copy the data from the host to the memory on the device.
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
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output based on the API.
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> scaleShape = {2};
    std::vector<int64_t> offsetShape = {2};
    std::vector<int64_t> outShape = {4, 2};
    void* selfDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> scaleHostData = {1, 2};
    std::vector<float> offsetHostData = {1, 2};
    std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
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

    const int32_t dstType = 2;
    const int32_t axis = -1;
    bool sqrtMode = false;
    const char* roundMode = "round";

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAscendQuantV3.
    ret = aclnnAscendQuantV3GetWorkspaceSize(
        self, scale, offset, sqrtMode, roundMode, dstType, axis, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAscendQuantV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnAscendQuantV3.
    ret = aclnnAscendQuantV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAscendQuantV3 failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value, copy the result from the device memory to the host, modify the configuration based on the API definition, and view resultData.
    auto size = GetShapeSize(outShape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int8_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. Release aclTensor. Modify the configuration based on the API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
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
