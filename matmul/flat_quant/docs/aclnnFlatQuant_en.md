# aclnnFlatQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:52:44.837Z -->

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

- **Description**: This fused operator performs two small matrix multiplications on the input matrix x at a time, that is, right-multiplying the input matrix by kroneckerP2 and left-multiplying it by kroneckerP1, and then quantizes the matrix multiplication result. Currently, the pertoken and pergroup quantization modes are supported, corresponding to the INT4 and FLOAT4_E2M1 quantized output types respectively.

- Matrix multiplication formula:

  1. Right-multiply the input x by kroneckerP2:

     $$
     x' = x @ kroneckerP2
     $$

  2. Left-multiply x' by kroneckerP1:

     $$
     x'' = kroneckerP1@x'
     $$

- Quantization calculation mode:

  pertoken quantization mode:

  1. Calculate the maximum absolute value along dimension 0 of x'' and divide it by (7 / clipRatio) to obtain the quantization scale for quantization to the INT4 format:

     $$
     quantScale = [max(abs(x''[0,:,:])),max(abs(x''[1,:,:])),...,max(abs(x''[M-1,:,:]))]/(7 / clipRatio)
     $$

  2. Calculate the output out:

     $$
     out = x'' / quantScale
     $$

  pergroup quantization mode

  1. After matrix multiplication, x'' has the shape [M,N1,N2]. To compute mx_quantize in the pergroup quantization mode, reshape it to [M,N1*N2], denoted as x2

  2. Divide the second dimension of x2 into groups by groupsize. Each group contains elements e0, e1...e31. emax is the exponent of the largest normal number of the target quantization format (FLOAT4_E2M1), which is 2 (the largest normal number of FLOAT4_E2M1 is 6.0, i.e., 1.5×2^2).

  3. Calculate sharedExp, that is, the exponent of the largest absolute value in the group minus emax:

     $$
     sharedExp[M,N1*N2/32] = floor(log2(max(|e0|,|e1|,....,|e31|))) - emax
     $$

  4. Calculate quantScale

     $$
     quantScale[M,N1*N2/32] = 2 ^ {sharedExp[M,N1*N2/32]}
     $$

  5. quantScale is shared within each groupsize. Calculate out

     $$
     out = x2 / quantScale
     $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the `aclnnFlatQuantGetWorkspaceSize` API first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the `aclnnFlatQuant` API to perform the computation.

```Cpp
aclnnStatus aclnnFlatQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *kroneckerP1,
  const aclTensor *kroneckerP2,
  double           clipRatio,
  aclTensor       *out,
  aclTensor       *quantScale,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFlatQuant(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnFlatQuantGetWorkspaceSize

- **Parameters**:
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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Raw input data, corresponding to `x` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>shape is [M, N1, N2], where M is not greater than 262144, and N1 and N2 are not greater than 256.</li><li>If the data type of out is INT32, N2 must be a multiple of 8.</li><li>If the data type of out is INT4, N2 must be an even number.</li><li>If the data type of out is FLOAT4_E2M1, N2 must be an even number.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kroneckerP1 (aclTensor*)</td>
      <td>Input</td>
      <td>Input computation matrix 1, corresponding to `kroneckerP1` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape is [N1, N1], where N1 is the same as the N1 dimension of x.</li><li>The data type is the same as that of x.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kroneckerP2 (aclTensor*)</td>
      <td>Input</td>
      <td>Calculation matrix 2 for the input, corresponding to `kroneckerP2` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape is [N2, N2], where N2 is the same as the N2 dimension of x. The shape [0,0] is supported only in perGroup mode, indicating that the operator does not use the right matrix for calculation and only performs multiplication of the left matrix.</li><li>The data type is the same as that of x.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>clipRatio (double)</td>
      <td>Input</td>
      <td>Controls the clipping ratio of quantization, corresponding to `clipRatio` in the corresponding formula.</td>
      <td><ul><li>Input value range: (0, 1].</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out(aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor, corresponding to out in the corresponding formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>If the data type is INT32, the last dimension of the shape is 1/8 of the last dimension of input x, and the other dimensions are the same as those of x.</li><li>If the data type is INT4, the shape is the same as that of input x.</li><li>If the data type is FLOAT4_E2M1, the shape is [M,N1*N2].</li></ul></td>
      <td>INT4, INT32, FLOAT4_E2M1</td>
      <td>ND</td>
      <td>3 or 2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantScale(aclTensor*)</td>
      <td>Output</td>
      <td>Quantization scale of the output, which is quantScale in the corresponding formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>If the quantized output type is INT4 or INT32, the shape is [M], where M is the same as the M dimension in x. The data type is FLOAT32.</li><li>If the quantized output type is FLOAT4_E2M1, the shape is [M,ceildiv(N1*N2,64),2]. The data type is FLOAT8_E8M0.</li></ul></td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1 or 3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that the user needs to allocate on the Device.</td>
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
      <td>Required input parameters (x, kroneckerP1, kroneckerP2) and required output parameters (out, quantScale) are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>The data type or data format of x, kroneckerP1, kroneckerP2, out, and quantScale is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of kroneckerP1 and kroneckerP2 are inconsistent with that of x.</td>
    </tr>
     <tr>
      <td>The dimension of x is not 3.</td>
    </tr>
    <tr>
      <td>The first dimension of x exceeds the range [1, 262144], the second dimension exceeds [1, 256], or the third dimension exceeds [1, 256].</td>
    </tr>
    <tr>
      <td>The dimension of kroneckerP1 is not 2, or the first and second dimensions are inconsistent with the second dimension of x.</td>
    </tr>
    <tr>
      <td>In the perToken case, the dimension of kroneckerP2 is not 2, or the first and second dimensions are inconsistent with the third dimension of x.</td>
    </tr>
    <tr>
      <td>In the INT4 or INT32 case, the dimension of quantScale is not 1, or the first dimension is inconsistent with the first dimension of x.</td>
    </tr>
    <tr>
      <td>In the FLOAT4_E2M1 case, the dimension of quantScale is not 3, or the first dimension is inconsistent with the first dimension of x.</td>
    </tr>
    <tr>
      <td>The value of clipRatio is out of the range (0, 1].</td>
    </tr>
    <tr>
      <td>When the data type of out is INT4, the trailing axis size of x is not an even number, or the shape of x is inconsistent with the shape of out.</td>
    </tr>
    <tr>
      <td>When the data type of out is INT32, the trailing axis size of x is not 8 times that of out, or the non-trailing axis sizes of x and out are inconsistent.</td>
    </tr>
  </tbody></table>

## aclnnFlatQuant

- **Parameters**:

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
      <td>Address of the workspace memory allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the device, obtained by the first-phase API aclnnFlatQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream that executes the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnFlatQuant defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_flat_quant.h"

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
    // Boilerplate for resource initialization.
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

    // Call aclrtMemcpy to copy data from the host to the device memory.
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

int main()
{
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs as required by the API.
    std::vector<int64_t> xShape = {16, 16, 16};
    std::vector<int64_t> kroneckerP1Shape = {16, 16};
    std::vector<int64_t> kroneckerP2Shape = {16, 16};
    std::vector<int64_t> outShape = {16, 16, 2};
    std::vector<int64_t> quantScaleShape = {16};
    void* xDeviceAddr = nullptr;
    void* kroneckerP1DeviceAddr = nullptr;
    void* kroneckerP2DeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* quantScaleDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* kroneckerP1 = nullptr;
    aclTensor* kroneckerP2 = nullptr;
    aclTensor* out = nullptr;
    aclTensor* quantScale = nullptr;
    double clipRatio = 1.0;
    std::vector<aclFloat16> xHostData(16 * 16 * 16, aclFloatToFloat16(1));
    std::vector<aclFloat16> kroneckerP1HostData(16 * 16, aclFloatToFloat16(1));
    std::vector<aclFloat16> kroneckerP2HostData(16 * 16, aclFloatToFloat16(1));
    std::vector<int32_t> outHostData(16 * 16 * 2, 1);
    std::vector<float> quantScaleHostData(16, 0);
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the kroneckerP1 aclTensor.
    ret = CreateAclTensor(
        kroneckerP1HostData, kroneckerP1Shape, &kroneckerP1DeviceAddr, aclDataType::ACL_FLOAT16, &kroneckerP1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the kroneckerP2 aclTensor.
    ret = CreateAclTensor(
        kroneckerP2HostData, kroneckerP2Shape, &kroneckerP2DeviceAddr, aclDataType::ACL_FLOAT16, &kroneckerP2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the quantScale aclTensor.
    ret = CreateAclTensor(
        quantScaleHostData, quantScaleShape, &quantScaleDeviceAddr, aclDataType::ACL_FLOAT, &quantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnFlatQuant.
    ret = aclnnFlatQuantGetWorkspaceSize(
        x, kroneckerP1, kroneckerP2, clipRatio, out, quantScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlatQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Request device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnFlatQuant.
    ret = aclnnFlatQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlatQuant failed. ERROR: %d\n", ret); return ret);
    // 4. (boilerplate) Wait synchronously for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device side to the host side. Modify it based on the definition of the specific API.
    auto size = GetShapeSize(outShape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    auto quantScaleSize = GetShapeSize(quantScaleShape);
    std::vector<float> quantScaleResultData(quantScaleSize, 0);
    ret = aclrtMemcpy(
        quantScaleResultData.data(), quantScaleResultData.size() * sizeof(quantScaleResultData[0]),
        quantScaleDeviceAddr, quantScaleSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < quantScaleSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, quantScaleResultData[i]);
    }

    // 6. Release aclTensor. Modify based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(kroneckerP1);
    aclDestroyTensor(kroneckerP2);
    aclDestroyTensor(out);
    aclDestroyTensor(quantScale);

    // 7. Release device resources. Modify based on the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(kroneckerP1DeviceAddr);
    aclrtFree(kroneckerP2DeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(quantScaleDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
