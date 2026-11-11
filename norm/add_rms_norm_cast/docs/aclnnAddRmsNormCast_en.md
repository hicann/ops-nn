# aclnnAddRmsNormCast

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:40:55.511Z -->

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

- Description: The RmsNorm operator is a commonly used normalization operation in large models. The AddRmsNormCast operator fuses the Cast operator after AddRmsNorm to reduce data movement in and out.
- Formula:

  $$
  x_i=float(x1_{i})+float(x2_{i})
  $$

  $$
  y1Out=\operatorname{RmsNorm}(x_i)=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i *g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  $$
  y2Out=cast(y1Out)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the `aclnnAddRmsNormCastGetWorkspaceSize` API first to obtain the workspace size required for computation and the executor containing the computation flow, and then call the `aclnnAddRmsNormCast` API to execute the computation.

```Cpp
aclnnStatus aclnnAddRmsNormCastGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  double           epsilon,
  const aclTensor *y1Out,
  const aclTensor *y2Out,
  const aclTensor *rstdOut,
  const aclTensor *xOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddRmsNormCast(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddRmsNormCastGetWorkspaceSize

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
      <td>x1 (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the first input for the Add calculation. It corresponds to `x1` in the formula.</td>
      <td>Supports empty tensors.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2 (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the second input for the Add calculation. It corresponds to `x2` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape and data type must remain consistent with those of `x1`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the scaling factor (weight) of RmsNorm. Corresponds to `gamma` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type must remain consistent with that of `x1`.</li><li>The shape must remain consistent with the trailing dimensions of `x1`, which are the dimensions of `x1` to be normalized.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Value added to the denominator to ensure numerical stability. Refers to `epsilon` in the corresponding formula.</td>
      <td>Recommended value: 1e-6.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output data after normalization. Corresponds to `y1Out` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>shape and data format must remain consistent with input parameter `x1`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>y2Out(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output data after normalization and type conversion. Corresponds to `y2Out` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape, data format, and data type must remain consistent with those of input parameter `x1`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>rstdOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the reciprocal of the standard deviation after normalization. Corresponds to the reciprocal of `Rms(x)` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data format must remain consistent with input parameter `x1`.</li><li>The number of dimensions must remain consistent with input parameter `x1`. The dimensions that do not require norm (the leading dimensions obtained by subtracting the dimension of `gamma` from the dimension of `x1`) must remain consistent with the corresponding dimensions of `x1`, and the dimensions that require norm (the trailing dimensions, whose count is the same as the number of dimensions of `gamma`) must all be 1. Example of the relationship between the `rstdOut` shape and the `x1` and `gamma` shapes: if the `x1` shape is (2,3,4,8) and the `gamma` shape is (8), the `rstdOut` shape is (2,3,4,1); if the `x1` shape is (2,3,4,8) and the `gamma` shape is (4,8), the `rstdOut` shape is (2,3,1,1).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>xOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the result of the Add computation. Corresponds to `x` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape, data format, and data type must all remain consistent with input parameter `x1`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation process.</td>
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
      <td>Returned if a required input, output, or required attribute is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of the input or output is not in the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The input and output do not meet the requirements in the preceding parameter description.</td>
    </tr>
  </tbody></table>

## aclnnAddRmsNormCast

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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnAddRmsNormCastGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Dimension boundary description:

  The size of each dimension in the shape of x1, x2, gamma, y1Out, y2Out, rstdOut, and xOut is not greater than the maximum INT32 value of 2147483647.

- Boundary value scenario description:
  - Currently, the empty tensor use case is not supported when the total number of elements in non-Norm dimensions is greater than 0 and the total number of elements in the Norm dimension is 0.
  - When the input is Inf, the output is Inf.
  - When the input is NaN, the output is NaN.

- Deterministic computation:
  - aclnnAddRmsNormCast defaults to a deterministic implementation.

## Example

The sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm_cast.h"

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
    // Boilerplate, resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId); aclFinalize(); return ret);

    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for memory on the device side.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data on the host side to memory on the device side.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Device/stream initialization. See the acl API documentation.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs. Custom construction is required based on the API.
    std::vector<int64_t> xShape = {2, 16};
    std::vector<int64_t> gammaShape = {16};
    std::vector<int64_t> yShape = {2, 16};
    std::vector<int64_t> rstdShape = {2, 1};
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* y1DeviceAddr = nullptr;
    void* y2DeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* y1 = nullptr;
    aclTensor* y2 = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* x = nullptr;
    std::vector<short> x1HostData = {0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700};
    std::vector<short> x2HostData = {0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700};
    std::vector<short> gammaHostData = {0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                        0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700};
    std::vector<float> y1HostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
                                     0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<short> y2HostData = {0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                     0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700};
    std::vector<float> rstdHostData = {1, 2};
    std::vector<short> xHostData = {0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                    0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                    0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                                    0x0000, 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700};
    float epsilon = 1e-6;

    // Create the x1 aclTensor.
    ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the x2 aclTensor.
    ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gamma aclTensor.
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the y1 aclTensor.
    ret = CreateAclTensor(y1HostData, yShape, &y1DeviceAddr, aclDataType::ACL_FLOAT, &y1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the y2 aclTensor.
    ret = CreateAclTensor(y2HostData, yShape, &y2DeviceAddr, aclDataType::ACL_FLOAT16, &y2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the rstd aclTensor.
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Replace it with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAddRmsNormCast.
    ret = aclnnAddRmsNormCastGetWorkspaceSize(x1, x2, gamma, epsilon, y1, y2, rstd, x, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormCastGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnAddRmsNormCast.
    ret = aclnnAddRmsNormCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormCast failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values. Copy the results on the device side to the host side. Modify based on the definition of the specific API.
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), y1DeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("y1 result[%ld] is: %f\n", i, resultData[i]);
    }

    std::vector<uint16_t> resultData1(size, 0);
    ret = aclrtMemcpy(
        resultData1.data(), resultData1.size() * sizeof(resultData1[0]), y2DeviceAddr, size * sizeof(resultData1[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("y2 result[%ld] is: 0x%04x\n", i, resultData1[i]);
    }

    // 6. Release the aclTensor and aclScalar. Modify based on the definition of the specific API.
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(gamma);
    aclDestroyTensor(y1);
    aclDestroyTensor(y2);
    aclDestroyTensor(rstd);
    aclDestroyTensor(x);

    // 7. Release the device resources. Modify based on the definition of the specific API.
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(y2DeviceAddr);
    aclrtFree(y1DeviceAddr);
    aclrtFree(rstdDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
