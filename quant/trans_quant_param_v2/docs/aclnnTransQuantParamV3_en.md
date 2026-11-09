# aclnnTransQuantParamV3

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Function: Converts quantization scale data from FLOAT32 to the UINT64 or INT64 type required by the hardware. Compared with [aclnnTransQuantParamV2](aclnnTransQuantParamV2_en.md), this interface adds the roundMode input to select the rounding mode used during data type conversion.
- Formula:

  1. `out` is in 64-bit format and is initialized to 0.

  2. If `roundMode` is 1, first round `scale` to a 19-bit floating-point representation (1 sign bit, 8 exponent bits, and 10 mantissa bits) using round to nearest, ties to even. Resolve ties to the value whose least significant mantissa bit is 0. Flush FLOAT32 subnormal values to zero while preserving the sign. Skip this step if `roundMode` is 0.

     $$
     scale = Round(scale)
     $$

  3. Reinterpret the processed `scale` bitwise as UINT32, denoted by `scaleBits`. Preserve its highest 19 bits (bits 13–31, with the lowest 13 bits cleared), write them into the lowest 32 bits of `out`, and set bit 46 to 1. Widen the value to UINT64 before bitwise operations. Bit positions are numbered from 0.

     $$
     out = out\ |\ (scaleBits\ \&\ 0xFFFFE000)\ |\ (1\ll46)
     $$

  4. Perform subsequent calculations based on the `offset` value:
     - If `offset` does not exist, no further computation is performed.
     - If `offset` exists:
       1. Round `offset` to the nearest integer, resolving ties to even, and clamp it to [-256, 255].

          $$
          offset = Max(Min(INT(Round(offset)),255),-256)
          $$

       2. Preserve the lowest 9 bits of `offset` and store them in bits 37–45 of `out`.

          $$
          out = (out\ \&\ 0x4000FFFFFFFF)\ |\ ((offset\ \&\ 0x1FF)\ll37)
          $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnTransQuantParamV3GetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnTransQuantParamV3" API to execute the computation.

```cpp
aclnnStatus aclnnTransQuantParamV3GetWorkspaceSize(
  const aclTensor* scale,
  const aclTensor* offset,
  int64_t          roundMode,
  const aclTensor* out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnTransQuantParamV3(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

## aclnnTransQuantParamV3GetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1500px">
    <colgroup>
      <col style="width: 210px">
      <col style="width: 100px">
      <col style="width: 240px">
      <col style="width: 390px">
      <col style="width: 230px">
      <col style="width: 120px">
      <col style="width: 110px">
      <col style="width: 100px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimensions (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>scale (const aclTensor*)</td>
        <td>Input</td>
        <td>Quantization scale value, corresponding to <code>scale</code> in the formula.</td>
        <td><ul><li>Empty tensors are not supported. NaN and Inf are not allowed.</li><li>Supports a one-dimensional shape (t,) or a two-dimensional shape (1,n), where t=1 or n. For two-dimensional shape restrictions in grouped scenarios, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1-2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>offset (const aclTensor*)</td>
        <td>Optional input</td>
        <td>Optional quantization offset value, corresponding to <code>offset</code> in the formula.</td>
        <td><ul><li>Empty tensors are not supported.</li><li>Supports a one-dimensional shape (t,) or a two-dimensional shape (1, n), where t=1 or n, and n matches the n dimension of the right matrix in matmul.</li><li>The user must ensure that offset contains no NaN or Inf.</li><li>If offset is unused, pass nullptr. Its bit field in the output remains 0.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1-2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>roundMode (int64_t)</td>
        <td>Input</td>
        <td>Rounding mode applied before encoding scale.</td>
        <td>0: clears the lowest 13 bits of the FLOAT32 representation, consistent with V2. 1: rounds to a 19-bit floating-point representation (1 sign bit, 8 exponent bits, and 10 mantissa bits) using round to nearest, ties to even. Other values are unsupported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (const aclTensor*)</td>
        <td>Output</td>
        <td>Quantization output, corresponding to <code>out</code> in the formula.</td>
        <td><ul><li>Empty tensors are not supported.</li><li>Without offset, the shape matches that of scale. With offset, the shape must match the broadcast result. See the shape description below for common combinations.</li></ul></td>
        <td>UINT64, INT64</td>
        <td>ND</td>
        <td>1-2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to allocate on the device, in bytes.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation process.</td>
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

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 300px">
      <col style="width: 150px">
      <col style="width: 550px">
    </colgroup>
    <thead>
      <tr>
        <th>Return/Status Code</th>
        <th>Value</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>The input scale or out is a null pointer.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td><ul><li>The data type or format of scale, offset, or out is unsupported.</li><li>The shape of scale or offset is unsupported.</li><li>The value of roundMode is unsupported.</li></ul></td>
      </tr>
    </tbody>
  </table>

## aclnnTransQuantParamV3

- **Parameters**

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 220px">
      <col style="width: 100px">
      <col style="width: 680px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>workspace (void*)</td>
        <td>Input</td>
        <td>Workspace memory address requested on the Device side.</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t)</td>
        <td>Input</td>
        <td>Workspace size to allocate on the Device, in bytes, obtained from the first-stage interface aclnnTransQuantParamV3GetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor*)</td>
        <td>Input</td>
        <td>Op executor that contains the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream (const aclrtStream)</td>
        <td>Input</td>
        <td>Specifies the stream for task execution.</td>
      </tr>
    </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

<!-- npu="950,A3,910b,310p" id7 -->
- <term>Atlas inference products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: This API can be used together with matmul operators (such as [aclnnQuantMatmulV4](../../../matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4_en.md)).
<!-- end id7 -->
<!-- npu="A3,910b,310p" id8 -->
- <term>Atlas inference products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: This API does not support use with grouped matmul class operators (such as aclnnGroupedMatmulV4).
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950PR&950DT products</term>: grouped scenarios without offset also support scale with shape (g,n), where g and n are positive integers. The shape of out matches that of scale.
<!-- end id9 -->
- The shapes of scale, offset, and out are described as follows:
  - When there is no offset, the out shape is the same as the scale shape.
    - If out is used as the input of a matmul operator (such as [aclnnQuantMatmulV4](../../../matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4_en.md)), the shape can be 1D (1,), (n,) or 2D (1, n), where n is the same as the shape n of the right matrix (corresponding to parameter x2) in matmul computation.
    <!-- npu="950" id10 -->
    - If out is used as a grouped matmul class operator input (such as aclnnGroupedMatmulV4), it is only used when the grouping mode is m-axis grouping (corresponding parameter groupType is 0). Its shape supports 1D (g,) or 2D (g, 1), (g, n), where n is consistent with the shape n of the right matrix (corresponding parameter x2) in the grouped matmul computation, and g is consistent with the number of groups (corresponding parameter groupListOptional's shape size) in the grouped matmul computation.
    <!-- end id10 -->
  - When offset is present, it can only be used as the input of a matmul operator (such as [aclnnQuantMatmulV4](../../../matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4_en.md)):
    - The shapes of offset, scale, and out support 1D (1,), (n,) or 2D (1, n), where n is consistent with the shape n of the right matrix (corresponding parameter x2) in the matmul computation.
    - When scale and offset are both one-dimensional, out is one-dimensional and its element count is the maximum of their element counts. If both input element counts exceed 1, they must be equal.
    - When scale and offset are both two-dimensional, out has shape (1,n), matching the broadcast result. Use the same number of dimensions for scale and offset where possible, avoiding mixed (n,) and (1,n) shapes.
- Deterministic computation:
  - aclnnTransQuantParamV3 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <memory>
#include <cstdint>
#include <cstdio>
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
    // Boilerplate code for resource initialization
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
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
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

    // 2. Construct the input and output. Customize the construction based on the API.
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
    // Create the scale aclTensor.
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the offset aclTensor.
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> offsetTensorPtr(offset, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_UINT64, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API and change it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // Call the first-phase API aclnnTransQuantParamV3.
    ret = aclnnTransQuantParamV3GetWorkspaceSize(scale, offset, roundMode, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV3GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second-phase API aclnnTransQuantParamV3.
    ret = aclnnTransQuantParamV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV3 failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Wait synchronously for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values, copy the results from device memory to host memory, and modify the code based on the specific API definition.
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
    // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = aclnnTransQuantParamV3Test(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV3Test failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
