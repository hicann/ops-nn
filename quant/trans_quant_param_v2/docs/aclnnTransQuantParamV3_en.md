# aclnnTransQuantParamV3

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/quant/trans_quant_param_v2)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    √    |
| <term>Atlas training series products</term>                             |    ×    |

## Function

- Description: Converts the data type of the quantization parameter **scale** from FLOAT32 to UINT64 or INT64 required by the hardware. Compared with aclnnTransQuantParamV2, the roundMode input is added to this API for selecting the mode for converting the data value during data type conversion.
- Formula:

  1. `out` is in 64-bit format and is initialized to 0.

  2. If `round_mode` is 1, `scale` is rounded to the higher 19 bits. If `round_mode` is 0, no processing is performed.

     $$
     scale = Round(scale)
     $$

  3. The higher 19 bits of `scale` are truncated and stored in bit 32 of `out`, and bit 46 is changed to 1.

     $$
     out = out\ |\ (scale\ \&\ 0xFFFFE000)\ |\ (1\ll46)
     $$

  4. The subsequent computation is performed based on the value of `offset`.
     - If `offset` does not exist, no subsequent computation is performed.
     - If `offset` exists:
       1. The value of `offset` is converted to an int value in the range of [–256, 255].

          $$
          offset = Max(Min(INT(Round(offset)),255),-256)
          $$

       2. Nine bits of `offset` are retained and stored in bits 37 to 45 of out.

          $$
          out = (out\ \&\ 0x4000FFFFFFFF)\ |\ ((offset\ \&\ 0x1FF)\ll37)
          $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnTransQuantParamV3GetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnTransQuantParamV3** is called to perform computation.

```Cpp
aclnnStatus aclnnTransQuantParamV3GetWorkspaceSize(
  const aclTensor* scale,
  const aclTensor* offset,
  int64_t          roundMode,
  const aclTensor* out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnTransQuantParamV3(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnTransQuantParamV3GetWorkspaceSize

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
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>scale value for quantization. It corresponds to `scale` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape is one-dimensional (t,), where t = 1 or n, or two-dimensional (1, n), where n is the same as the shape n of the right matrix in the matmul computation. </li><li>Ensure that NaN and inf do not exist in the data of scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td>(Optional) offset value for quantization. It corresponds to `offset` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape is one-dimensional (t,) or two-dimensional (1, n), where t = 1 or n, and n is the same as the shape n of the right matrix in the matmul computation. </li><li>Ensure that NaN and inf do not exist in the data of offset.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>roundMode</td>
      <td>Input</td>
      <td>Round mode for packing FP32 into FP19 during quantization. It corresponds to `roundMode` in the formula.</td>
      <td>Only the following values are supported: 0 (compatible with V2) and 1 (improving compute precision).</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Compute output for quantization. It corresponds to `out` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>When the shape of the input scale is one-dimensional, the shape of out is also one-dimensional, and the shape size is the larger of the two one-dimensional shape sizes of scale and offset (if it is not nullptr). When the shape of the input scale is two-dimensional, the shape of out is the same as that of the input scale in terms of dimension and size.</li></ul></td>
      <td>UINT64, INT64</td>
      <td>ND</td>
      <td>1-2</td>
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
  
- **Returns:**

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
      <td>The passed scale or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type or format of scale, offset, or out is not supported.</td>
    </tr>
    <tr>
      <td>The shape of offset or scale is not (t,) or (1, n), where t = 1 or n, and n is the same as the shape n of the right matrix in the matmul computation.</td>
    </tr>
    <tr>
      <td>The value of roundMode is not supported.</tr>
  </tbody></table>

## aclnnTransQuantParamV3

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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnTransQuantParamV3GetWorkspaceSize.</td>
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

- <term>Atlas inference series products</term>, <term>Atlas A2 training series products/Atlas A2 inference series products</term>, and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: This API can be used together with matmul operators (such as [aclnnQuantMatmulV4](../../../matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4_en.md)).
- <term>Atlas inference series products</term>, <term>Atlas A2 training series products/Atlas A2 inference series products</term>, and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: This API cannot be used together with grouped matmul operators (such as aclnnGroupedMatmulV4).
- The shapes of **scale**, **offset**, and **out** are described as follows:
  - If **offset** does not exist, the shape of **out** is the same as that of **scale**.
    - If **out** is used as the input of a matmul operator (for example, [aclnnQuantMatmulV4](../../../matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4_en.md)), the shape can be one-dimensional (1,) or (n,), or two-dimensional (1, n). Here, **n** is the same as the shape of the right matrix (corresponding to **x2**) in matmul computation.
    - If **out** is used as the input of a grouped matmul operator (for example, aclnnGroupedMatmulV4), it is used only when the grouping mode is m-axis grouping (corresponding to groupType = 0). The shape can be one-dimensional (g,), or two-dimensional (g, 1), or (g, n). Here, **n** is the same as the shape of the right matrix (corresponding to **x2**) in grouped matmul computation, and **g** is the same as the number of groups (corresponding to the shape size of **groupListOptional**) in grouped matmul computation.
  - If **offset** exists, it is used only as the input of a matmul operator (for example, [aclnnQuantMatmulV4](../../../matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4_en.md)).
    - The shapes of **offset**, **scale**, and **out** can be one-dimensional (1,) or (n,), or two-dimensional (1, n). Here, **n** is the same as the shape of the right matrix (corresponding to **x2**) in matmul computation.
    - If the shape of the input **scale** is one-dimensional, the shape of **out** is also one-dimensional, and the shape size is the larger of the two one-dimensional shape sizes of **scale** and **offset**.
    - If the shape of the input **scale** is two-dimensional, the shape of **out** is the same as that of the input **scale** in terms of dimension and size.
- Deterministic compute:
  - **aclnnTransQuantParamV3** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <memory>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_trans_quant_param_v3.h"

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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnTransQuantParamV3Test(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output based on the API definition.
    std::vector<int64_t> offsetShape = {3};
    std::vector<int64_t> scaleShape = {3};
    std::vector<int64_t> outShape = {3};
    void* scaleDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> scaleHostData = {1, 1, 1};
    std::vector<float> offsetHostData = {1, 1, 1};
    std::vector<uint64_t> outHostData = {1, 1, 1};
    int64_t roundMode = 1;
    // Create a scale aclTensor.
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an offset aclTensor.
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> offsetTensorPtr(offset, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_UINT64, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnTransQuantParamV3.
    ret = aclnnTransQuantParamV3GetWorkspaceSize(scale, offset, roundMode, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV3GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second-phase API of aclnnTransQuantParamV3.
    ret = aclnnTransQuantParamV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV3 failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<uint64_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %lu\n", i, resultData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnTransQuantParamV3Test(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV3Test failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
