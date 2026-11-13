# aclnnAntiMxQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:04:12.217Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: Dequantizes the FLOAT4/FLOAT8 tensor quantized by calling `aclnnDynamicMxQuant/aclnnDynamicMxQuantV2` to the FLOAT16/BFLOAT16/FLOAT32 format.

- Formula:

  $$
  X_{dq} = X_q \times 2^{sf - bias}
  $$

  - where $sf$ is the scaling factor provided by the input mxscale; $bias$ is the exponent bias. For the FLOAT8_E8M0 format, $bias = 127$; $X_q$ is the quantized FLOAT4/FLOAT8 tensor; $X_{dq}$ is the dequantized FLOAT16/BFLOAT16/FLOAT32 tensor.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the `aclnnAntiMxQuantGetWorkspaceSize` API first to obtain the workspace size required for computation and the executor that contains the computation process of the operator, and then call the `aclnnAntiMxQuant` API to perform the computation.

```cpp
aclnnStatus aclnnAntiMxQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *mxscale,
  int64_t          axis,
  int64_t          dstType,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnAntiMxQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAntiMxQuantGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 300px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 250px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Shape (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Quantization tensor input of the operator. Corresponds to Xq in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>If x is an empty tensor, y and mxscale must also be empty tensors.</li></ul></td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>1-7, must be consistent with the output of DynamicMxQuant</td>
      <td>FLOAT8 supports non-contiguous tensors, FLOAT4 does not support non-contiguous tensors</td>
    </tr>
    <tr>
      <td>mxscale (aclTensor*)</td>
      <td>Input</td>
      <td>Quantization scale computed by calling DynamicMxQuant. Corresponds to sf in the formula.</td>
      <td><ul><li>Empty tensors are allowed only when x is an empty tensor. Otherwise, empty tensors (shapes containing a 0 dimension) are not supported and 161002 is returned.</li><li>The shape must satisfy the formula in Constraints.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-8, must be consistent with the output of DynamicMxQuant</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis (int64_t)</td>
      <td>Input</td>
      <td>Specifies the dequantization axis.</td>
      <td>Value range: [-D, D-1], where D is the number of dimensions of input x.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Specifies the data type of output y.</td>
      <td>Value range: {0, 1, 27}, corresponding to the data type of output y: {0: FLOAT32, 1: FLOAT16, 27: BFLOAT16}.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the dequantized output tensor. Corresponds to Xdq in the corresponding formula.</td>
      <td><ul><li>Empty tensors are allowed only when x is an empty tensor. Otherwise, empty tensors are not supported.</li><li>The shape must be exactly the same as that of x.</li><li>The data type must correspond to dstType.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the computation process of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 319px">
  <col style="width: 108px">
  <col style="width: 621px">
  </colgroup>
  <thead>
  <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
  </tr></thead>
  <tbody>
  <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>The parameter is a required input or output and is a null pointer.</td>
  </tr>
  <tr>
  </tr>
  <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type of x, mxscale, or y is not within the supported range.</td>
  </tr>
  <tr>
    <td>The shape of x, mxscale, or y does not meet the check requirements.</td>
  </tr>
  <tr>
    <td>The dimension of x, mxscale, or y is not within the supported range.</td>
  </tr>
  <tr>
    <td>The value of axis or dstType is not currently supported.</td>
  </tr>
  <tr>
    <td>The data type of y does not satisfy the corresponding relationship with dstType.</td>
  </tr>
  <tr>
    <td>Only some of x, mxscale, and y are empty tensors.</td>
  </tr>
  <tr>
    <td rowspan="1">ACLNN_ERR_RUNTIME_ERROR</td>
    <td rowspan="1">361001</td>
    <td>The current platform is not within the range of supported platforms.</td>
  </tr>
  </tbody></table>

## aclnnAntiMxQuant

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 173px">
  <col style="width: 127px">
  <col style="width: 748px">
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
      <td>Pointer to the workspace memory allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the device, obtained by calling the first-phase API aclnnAntiMxQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the computation process of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - `aclnnAntiMxQuant` defaults to a deterministic implementation.
- Constraints on the shapes of x and mxscale are as follows:
  - If the data type of input x is float4_e2m1 or float4_e1m2, x.shape[-1] must be an even number.
  - axis_change = axis if axis >= 0 else axis + rank(x).
  - mxscale.shape[axis_change] = (ceil(x.shape[axis] / 32) + 2 - 1) / 2.
  - mxscale.shape[-1] = 2.
  - rank(mxscale) = rank(x) + 1.
  - Other dimensions are the same as those of input x.
- Empty tensor constraint: only the scenario where x, y, and mxscale are all empty tensors is supported. In this case, the API returns success and workspaceSize is 0. If only some of the inputs are empty tensors (for example, mxscale contains a 0 dimension in its shape while x is not empty), the input is invalid and 161002 is returned.

## Example

The sample code is as follows and is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_anti_mx_quant.h"

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
    // Boilerplate code, resource initialization
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
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
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

int aclnnAntiMxQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs. Customize the construction based on the API.
    // x: FP8_E4M3FN, shape {1, 4}
    std::vector<int64_t> xShape = {1, 4};
    // mxscale: FP8_E8M0, shape {1, 1, 2}
    // ceil(4/32) = 1, (1+2-1)/2 = 1, append 2 -> {1, 1, 2}
    std::vector<int64_t> mxscaleShape = {1, 1, 2};
    // y: BF16, shape {1, 4}
    std::vector<int64_t> yOutShape = {1, 4};

    void* xDeviceAddr = nullptr;
    void* mxscaleDeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* mxscale = nullptr;
    aclTensor* yOut = nullptr;

    std::vector<uint8_t> xHostData = {0, 72, 96, 120};
    std::vector<uint8_t> mxscaleHostData = {128, 0};
    std::vector<uint16_t> yOutHostData = {0, 16640, 17024, 17408};

    int64_t axis = -1;
    int64_t dstType = 27;  // BFLOAT16

    // Create the x aclTensor(FP8_E4M3FN).
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the mxscale aclTensor(FP8_E8M0).
    ret = CreateAclTensor(mxscaleHostData, mxscaleShape, &mxscaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscaleTensorPtr(mxscale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscaleDeviceAddrPtr(mxscaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the yOut aclTensor(BF16).
    ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_BF16, &yOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnAntiMxQuant.
    ret = aclnnAntiMxQuantGetWorkspaceSize(x, mxscale, axis, dstType, yOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAntiMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // Call the second-phase API of aclnnAntiMxQuant.
    ret = aclnnAntiMxQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAntiMxQuant failed. ERROR: %d\n", ret); return ret);

    // Wait for the task execution to complete synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Obtain the value of output y and copy the result from the device memory to the host.
    auto size = GetShapeSize(yOutShape);
    std::vector<uint16_t> yOutData(size, 0);
    ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                      size * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("y[%ld] is: %u\n", i, yOutData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. (Boilerplate) Device/stream initialization. Refer to the ACL API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnAntiMxQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAntiMxQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
