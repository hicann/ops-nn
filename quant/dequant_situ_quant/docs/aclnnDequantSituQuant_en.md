# aclnnDequantSituQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:08:10.698Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

- Description: Adds dequant and quant operations before and after the Situ activation function to implement DequantSituQuant computation of x.

- Formula:

1. The dequantization path varies depending on the data type of the input x:

   - INT8 path

     $$
     dequantOut_i = cast\_to\_float(x_i) \times weight\_scale_i + bias_i
     $$

   - INT32 path

     $$
     dequantOut_i = cast\_to\_float(x_i) \times weight\_scale_i \times activation\_scale_i + bias_i
     $$

   - BF16/FLOAT16 path (pre-dequantization)

    $$
    dequantOut_i = cast\_to\_float(x_i)
    $$

2. Situ activation

   $$
   situ_a = \beta \times \tanh(gate / \beta) \times sigmoid(gate)
   $$

   When linear_beta > 0:

   $$
   up = linear\_beta \times \tanh(up / linear\_beta)
   $$

   $$
   situOut = situ_a \times up
   $$

    Where activate_left is true, gate takes the first half of dequantOut and up takes the second half; where activate_left is false, gate takes the second half of dequantOut and up takes the first half.

3. Quantization

   - quant_type = static

     $$
     y_i = trunc(situOut_i / quant\_scale_i + quant\_offset_i)
     $$

   - quant_type = dynamic

    $$
    scale_i = absmax(situOut_i) / 127
    $$

    $$
    y_i = trunc(situOut_i / scale_i)
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnDequantSituQuantGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnDequantSituQuant" to execute the computation.

```Cpp
aclnnStatus aclnnDequantSituQuantGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weightScaleOptional,
    const aclTensor* activationScaleOptional,
    const aclTensor* biasOptional,
    const aclTensor* quantScaleOptional,
    const aclTensor* quantOffsetOptional,
    const aclTensor* groupIndexOptional,
    double            beta,
    double            linearBeta,
    bool              activateLeft,
    char*             quantTypeOptional,
    const aclTensor*  yOut,
    const aclTensor*  yScaleOut,
    uint64_t*         workspaceSize,
    aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnDequantSituQuant(
    void*            workspace,
    uint64_t         workspaceSize,
    aclOpExecutor*   executor,
    aclrtStream      stream)
```

## aclnnDequantSituQuantGetWorkspaceSize

- **Parameters:**

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
      <td>Input x in the formula.</td>
      <td><ul><li>Does not support null tensors.</li><li>If x is of the INT8 data type, the dimension of x must be ≥2; if x is of the INT32/BF16/FLOAT16 data type, the dimension of x must be 2.</li><li>The last dimension must be a multiple of 2.</li></ul></td>
      <td>INT8, INT32, BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>x</td>
    </tr>
    <tr>
      <td>weightScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Weight scale for dequantization.</td>
      <td><ul><li>Does not support null Tensor.</li><li>The shape is (2H,) or (1,).</li><li>Required for INT8 and INT32; not used for BF16/FLOAT16.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>activationScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Activation scale for dequantization.</td>
      <td><ul><li>Does not support null Tensor.</li><li>The shape is (1,).</li><li>Required only for INT32; not used for other data types.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensor*)</td>
      <td>Input</td>
      <td>The bias for dequantization.</td>
      <td><ul><li>Does not support null tensor.</li><li>The shape is (2H,) or (1,).</li><li>Optional for INT8 and INT32, not used for BF16/FLOAT16.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>quantScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Scale for quantization.</td>
      <td><ul><li>Does not support null tensors.</li><li>Required when `quantTypeOptional` is static, with shape (H,) or (1,).</li><li>Optional when `quantTypeOptional` is dynamic; used as smoothScale.</li><li>Used for INT8 only.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>quantOffsetOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Quantized offset.</td>
      <td><ul><li>Does not support null tensor.</li><li>Optional when `quantTypeOptional` is static (treated as 0 if absent), with the same shape as `quantScaleOptional`.</li><li>Not used when `quantTypeOptional` is dynamic; must be empty.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input</td>
      <td>group_index of the MoE group.</td>
      <td><ul><li>Does not support null Tensor.</li><li>INT32/BF16/FLOAT16 optional, INT8 not used.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>beta (double)</td>
      <td>Input</td>
      <td>Beta parameter for Situ activation.</td>
      <td>Cannot be 0. Defaults to 4.0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linearBeta (double)</td>
      <td>Input</td>
      <td>The linear_beta parameter of the Situ activation.</td>
      <td>Not enabled when the value is ≤ 0. The default value is 25.0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateLeft (bool)</td>
      <td>Input</td>
      <td>Whether to apply Situ activation to the left half of the input.</td>
      <td>If the value is false, activation is applied to the right half of the input. Defaults to true.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantTypeOptional (char*)</td>
      <td>Input</td>
      <td>Quantization mode.</td>
      <td>Only {"static", "dynamic"} is supported. The default is "dynamic".</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Quantized output.</td>
      <td><ul><li>Does not support null tensors.</li><li>INT8: shape is x.shape[:-1]+[H]; others: shape is [M, H].</li></ul></td>
      <td>INT8</td>
      <td>ND</td>
      <td>2-8</td>
      <td>x</td>
    </tr>
    <tr>
      <td>yScaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Scale for dynamic quantization.</td>
      <td><ul><li>Does not support null tensors.</li><li>INT8: shape is x.shape[:-1]; others: shape is [M].</li><li>When `quantTypeOptional` is static, the output contains meaningless values.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-7</td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device.</td>
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

- **Return value**

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

The first-phase API performs input parameter validation and returns an error in the following scenarios:

<table style="undefined;table-layout: fixed;width: 1048px"><colgroup>
<col style="width: 319px">
<col style="width: 108px">
<col style="width: 621px">
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
    <td rowspan="4">ACLNN_ERR_PARAM_NULLPTR</td>
    <td rowspan="4">161001</td>
    <td>x, yOut, or yScaleOut is a null pointer.</td>
  </tr>
  <tr>
    <td>weightScaleOptional is a null pointer when the data type is INT8.</td>
  </tr>
  <tr>
    <td>weightScaleOptional or activationScaleOptional is a null pointer when the data type is INT32.</td>
  </tr>
  <tr>
    <td>quantScaleOptional is a null pointer when `quantTypeOptional` is static.</td>
  </tr>
  <tr>
    <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="6">161002</td>
    <td>x, yOut, or yScaleOut is an empty tensor.</td>
  </tr>
  <tr>
    <td>The data types of x, weightScaleOptional, and other parameters are not within the supported range.</td>
  </tr>
  <tr>
    <td>quantTypeOptional is not within the supported range.</td>
  </tr>
  <tr>
    <td>The last dimension of x is not a multiple of 2.</td>
  </tr>
  <tr>
    <td>beta is 0.</td>
  </tr>
  <tr>
    <td>The input constraints are not met.</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_RUNTIME_ERROR</td>
    <td>361001</td>
    <td>The current platform is not within the range of supported platforms.</td>
  </tr>
</tbody></table>

## aclnnDequantSituQuant

- **Parameters:**
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
      <td>Workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device side, obtained by the first-phase API aclnnDequantSituQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnDequantSituQuant defaults to a deterministic implementation.

- The last dimension of x must be a multiple of 2.
- When the data type of x is INT8, x must have at least 2 dimensions; when the data type of x is INT32/BF16/FLOAT16, x must have 2 dimensions.
- beta cannot be 0.
- INT8: `weightScaleOptional` is required, while `activationScaleOptional` and `groupIndexOptional` are forbidden.
- INT32: `weightScaleOptional` and `activationScaleOptional` are required, while `quantScaleOptional` and `quantOffsetOptional` are forbidden.
- BF16/FLOAT16: None of the optional inputs is used (pre-dequantization mode).
- When `quantTypeOptional` is static, `quantScaleOptional` is required.
- When `quantTypeOptional` is static, `quantOffsetOptional` is optional and treated as 0 when absent; when
  `quantTypeOptional` is dynamic, `quantOffsetOptional` must be empty (not used).
- When `quantTypeOptional` is dynamic, `quantScaleOptional` is optional (used as smoothScale).

## Example

The sample code is as follows and is for reference only. For details about how to compile and run the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_dequant_situ_quant.h"

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
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Custom construction is required based on the API definition.
    std::vector<int64_t> xShape = {16, 64};
    std::vector<int64_t> wsShape = {64};
    std::vector<int64_t> yShape = {16, 32};
    std::vector<int64_t> yScaleShape = {16};

    auto xSize = GetShapeSize(xShape);
    std::vector<int8_t> xHostData(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        xHostData[i] = static_cast<int8_t>((i * 7 + 3) % 100);
    }
    std::vector<float> wsHostData(64, 0.1f);
    std::vector<int8_t> yHostData(GetShapeSize(yShape), 0);
    std::vector<float> yScaleHostData(GetShapeSize(yScaleShape), 0.0f);

    void* xDeviceAddr = nullptr;
    void* wsDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* yScaleDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* ws = nullptr;
    aclTensor* y = nullptr;
    aclTensor* yScale = nullptr;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wsHostData, wsShape, &wsDeviceAddr, aclDataType::ACL_FLOAT, &ws);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT8, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yScaleHostData, yScaleShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT, &yScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double beta = 4.0;
    double linearBeta = 25.0;
    bool activateLeft = true;

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnDequantSituQuant.
    ret = aclnnDequantSituQuantGetWorkspaceSize(x, ws, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                 beta, linearBeta, activateLeft,
                                                 const_cast<char*>("dynamic"),
                                                 y, yScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSituQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnDequantSituQuant.
    ret = aclnnDequantSituQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSituQuant failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result on the device side to the host side. Modify it based on the definition of the specific API.
    auto yResultSize = GetShapeSize(yShape);
    std::vector<int8_t> yResult(yResultSize, 0);
    ret = aclrtMemcpy(yResult.data(), yResult.size() * sizeof(int8_t), yDeviceAddr,
                      yResultSize * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yResult from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < yResultSize; i++) {
        LOG_PRINT("y[%ld] is: %d\n", i, yResult[i]);
    }

    // 6. Release the aclTensor and aclScalar. Modify it based on the definition of the specific API.
    aclDestroyTensor(x);
    aclDestroyTensor(ws);
    aclDestroyTensor(y);
    aclDestroyTensor(yScale);

    // 7. Release the device resources. Modify it based on the definition of the specific API.
    aclrtFree(xDeviceAddr);
    aclrtFree(wsDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(yScaleDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
