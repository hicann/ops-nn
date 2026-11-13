# aclnnMxToBlockMxQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:37:38.648Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not Supported
<!-- end id6 -->

## Function

- Description: Converts the FLOAT4 tensor quantized by calling `aclnnDynamicMxQuantV2` together with the FLOAT8_E8M0 scaling factors to the FLOAT8 block quantization format, and outputs the quantization scales along the -1 and -2 axes.

- Formula:

  $$
  \begin{aligned}
  scale_{fp8} &= max(mxscale_{fp4\_block}) / 2^{MAX\_OFFSET} \\
  offset &= mxscale_{fp4\_block} / scale_{fp8} \\
  x_{fp8} &= x_{fp4} \times offset
  \end{aligned}
  $$

  - where $mxscale_{fp4\_block}$ is the FP8_E8M0 scaling factor provided by the input mxscale; $MAX\_OFFSET$ is the exponent offset between the input and output data types; $x_{fp4}$ is the quantized FLOAT4 tensor; and $x_{fp8}$ is the converted FLOAT8 tensor.

  - MAX_OFFSET reference table:

    | Input    | Output     | MAX_OFFSET |
    | -------- | ---------- | ---------- |
    | FP4_E2M1 | FP8_E5M2   | 13         |
    | FP4_E1M2 | FP8_E5M2   | 15         |
    | FP4_E2M1 | FP8_E4M3FN | 6          |
    | FP4_E1M2 | FP8_E4M3FN | 8          |

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the `aclnnMxToBlockMxQuantGetWorkspaceSize` API to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the `aclnnMxToBlockMxQuant` API to execute the computation.

```cpp
aclnnStatus aclnnMxToBlockMxQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *mxscale,
  int64_t          dstType,
  const aclTensor *y,
  const aclTensor *scale1,
  const aclTensor *scale2,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMxToBlockMxQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnMxToBlockMxQuantGetWorkspaceSize

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
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the quantized tensor input of the operator. Corresponds to x_fp4 in the corresponding formula.</td>
      <td><ul><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>mxscale (aclTensor*)</td>
      <td>Input</td>
      <td>The quantization scale calculated by calling DynamicMxQuant. Corresponds to mxscale_fp4_block in the corresponding formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape must satisfy the formulas in Constraints.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3-4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Specifies the data type of output y.</td>
      <td>The value range is {35, 36}, which respectively correspond to the data type of output y: {35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantization output tensor. Corresponds to x_fp8 in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape dimensions must be kept consistent with x.</li><li>The data type must correspond to dstType.</li></ul></td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scale1 (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantization scale corresponding to each group along the -1 axis.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape must satisfy the formula in Constraints.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3-4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scale2 (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantization scale corresponding to each group along the -2 axis. The output needs to be interleaved for every two rows of data.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape must satisfy the formula in Constraints.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3-4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
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
      <td>The input parameter is a required input or output and is a null pointer.</td>
  </tr>
  <tr>
  </tr>
  <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types of x, mxscale, y, scale1, and scale2 are not within the supported range.</td>
  </tr>
  <tr>
    <td>The shapes of x, mxscale, y, scale1, and scale2 do not meet the check conditions.</td>
  </tr>
  <tr>
    <td>The dimensions of x, mxscale, y, scale1, and scale2 are not within the supported range.</td>
  </tr>
  <tr>
    <td>dstType is not within the currently supported values.</td>
  </tr>
  <tr>
    <td>The data type of y does not conform to the corresponding relationship with dstType.</td>
  </tr>
  <tr>
    <td rowspan="1">ACLNN_ERR_RUNTIME_ERROR</td>
    <td rowspan="1">361001</td>
    <td>The current platform is not within the supported platform range.</td>
  </tr>
  </tbody></table>

## aclnnMxToBlockMxQuant

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
      <td>The workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size requested on the Device side, obtained by calling the first-phase API aclnnMxToBlockMxQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnMxToBlockMxQuant` defaults to a deterministic implementation.
- x supports only 2D or 3D input. The -2 axis must be a multiple of 64, and the -1 axis must be a multiple of 2. Non-contiguous tensors and empty tensors are not supported.
- The shape constraints of x, mxscale, scale1, and scale2 are described as follows:
  - rank(mxscale) = rank(x) + 1.
  - mxscale.shape[-2] = (Ceil(x.shape[-1] / 32) + 2 - 1) / 2.
  - mxscale.shape[-1] = 2.
  - Other dimensions are consistent with input x.
- The shape constraints of output scale1 are described as follows:
  - rank(scale1) = rank(x) + 1.
  - scale1.shape[-2] = (Ceil(x.shape[-1] / 32) + 2 - 1) / 2.
  - scale1.shape[-1] = 2.
  - Other dimensions are kept consistent with input x.
- The shape constraints of output scale2 are as follows:
  - rank(scale2) = rank(x) + 1.
  - scale2.shape[-3] = ((Ceil(x.shape[-2] / 32) + 2 - 1) / 2) * 2 / 2.
  - scale2.shape[-2] = x.shape[-1].
  - scale2.shape[-1] = 2.
  - Other dimensions are consistent with input x.
- dstType supports only {35, 36}, corresponding to {FLOAT8_E5M2, FLOAT8_E4M3FN}.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_mx_to_block_mx_quant.h"

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
    // Boilerplate: resource initialization
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
    // Call aclrtMalloc to request device-side memory.
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

int aclnnMxToBlockMxQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs as required by the API.
    std::vector<int64_t> xShape = {64, 2};
    std::vector<int64_t> scaleShape = {64, 1, 2};
    std::vector<int64_t> yOutShape = {64, 2};
    std::vector<int64_t> scale1Shape = {64, 1, 2};
    std::vector<int64_t> scale2Shape = {1, 2, 2};

    void* xDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    void* scale1DeviceAddr = nullptr;
    void* scale2DeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* mxscale = nullptr;
    aclTensor* yOut = nullptr;
    aclTensor* scale1Out = nullptr;
    aclTensor* scale2Out = nullptr;

    std::vector<uint8_t> xHostData(128, 0x12);
    std::vector<uint8_t> scaleHostData(128, 128);
    std::vector<uint8_t> yOutHostData(128, 0);
    std::vector<uint8_t> scale1HostData(128, 0);
    std::vector<uint8_t> scale2HostData(4, 0);
    int64_t dstType = 36;

    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the scale aclTensor.
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(mxscale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the yOut aclTensor.
    ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &yOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the scale1Out aclTensor.
    ret = CreateAclTensor(scale1HostData, scale1Shape, &scale1DeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scale1TensorPtr(scale1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scale1DeviceAddrPtr(scale1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the scale2Out aclTensor.
    ret = CreateAclTensor(scale2HostData, scale2Shape, &scale2DeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scale2TensorPtr(scale2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scale2DeviceAddrPtr(scale2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnMxToBlockMxQuant.
    ret = aclnnMxToBlockMxQuantGetWorkspaceSize(x, mxscale, dstType, yOut, scale1Out, scale2Out,
                                                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMxToBlockMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // Call the second-phase API of aclnnMxToBlockMxQuant.
    ret = aclnnMxToBlockMxQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMxToBlockMxQuant failed. ERROR: %d\n", ret); return ret);

    // Synchronize and wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Obtain the value of output y and copy the result on device-side memory to the host side.
    auto ySize = GetShapeSize(yOutShape);
    std::vector<uint8_t> yOutData(ySize, 0);
    ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                      ySize * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < ySize; i++) {
        LOG_PRINT("y[%ld] is: %u\n", i, yOutData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnMxToBlockMxQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMxToBlockMxQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
