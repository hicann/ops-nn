# aclnnRmsNormQuantV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:36:41.579Z -->

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

- Description: The RmsNorm operator is a normalization operation commonly used in large models. Compared with the LayerNorm operator, it removes the mean subtraction. The RmsNormQuantV3 operator fuses the RmsNorm operator and the Quantize operator after RmsNorm, reducing data movement operations. It also adds the Rstd output based on the RmsNormQuantV2 operator.
- Formula:

  $$
  quant\_in_i=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} gamma_i + beta_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - When divMode is true:

    $$
    y=round((quant\_in/scale)+offset)
    $$

  - When divMode is false:

    $$
    y=round((quant\_in*scale)+offset)
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnRmsNormQuantV3GetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation process. Then call the "aclnnRmsNormQuantV3" API to perform the computation.

```Cpp
aclnnStatus aclnnRmsNormQuantV3GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *gamma,
  const aclTensor *scale,
  const aclTensor *offsetOptional,
  const aclTensor *betaOptional,
  double           epsilon,
  bool             divMode,
  bool             outputRstd,
  aclTensor       *y,
  aclTensor       *rstd,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnRmsNormQuantV3(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnRmsNormQuantV3GetWorkspaceSize

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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Source data tensor in the normalization process. Corresponds to `x` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>For details, see Constraints.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Weight tensor in the normalization process. Corresponds to `gamma` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The data type must be the same as that of `x`.</li><li>If the shape is one-dimensional, it must be the same as the last dimension of `x`.</li><li>If the shape is two-dimensional, the first dimension must be 1, and the second dimension must be the same as the last dimension of `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale (aclTensor*)</td>
      <td>Input</td>
      <td>Scale tensor used to obtain y during quantization, corresponding to `scale` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The dimension is 1. The shape size is 1 or the last dimension size of `x`.</li><li>For data type constraints, see Constraints.</li><li>The value cannot be 0.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offsetOptional (aclTensor*)</td>
      <td>Optional input</td>
      <td>Offset tensor used to obtain y during quantization, corresponding to `offset` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Optional parameter. A null pointer can be passed.</li><li>The shape must be the same as that of `scale`.</li><li>For data type constraints, see Constraints.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>betaOptional (aclTensor*)</td>
      <td>Optional input</td>
      <td>Represents the offset tensor in the normalization process. Corresponds to `beta` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Optional parameter. A null pointer can be passed in.</li><li>The data type must be the same as that of `x`.</li><li>If the shape is one-dimensional, the shape must be the same as the last dimension of `x`.</li><li>If the shape is two-dimensional, the first dimension must be 1, and the second dimension must be the same as the last dimension of `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Input `epsilon` in the formula, used to prevent division by zero. Data type: DOUBLE.</td>
      <td>A small positive value is recommended.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divMode (bool)</td>
      <td>Input</td>
      <td>Parameter in the formula that determines whether the quantization formula uses division. Data type: bool.</td>
      <td>Supports True and False.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputRstd (bool)</td>
      <td>Input</td>
      <td>Specifies whether to output a valid rstdOut.</td>
      <td><ul><li>Supports True and False.</li><li>When outputRstd is False, rstdOut is an invalid output.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Final quantization output tensor, corresponding to `y` in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape must be the same as that of input `x`.</li></ul></td>
      <td>INT8, INT32, INT4, FLOAT8, HIF8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the reciprocal of the standard deviation after normalization. Corresponds to the reciprocal of Rms(x) in the corresponding formula.</td>
      <td><ul><li>When outputRstd is True, empty tensors are not supported. The shape must be the same as the first few dimensions of input x, that is, the dimensions of x minus 1, indicating the dimensions that do not require normalization. The -1 axis of rstdOut is 1.</li><li>When outputRstd is False, the final output of this parameter is invalid. An empty pointer or an empty tensor can be passed as a placeholder.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
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
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>x, gamma, scale, or y is a null pointer.</td>
    </tr>
    <tr>
      <td>When outputRstd is True, rstd is a null pointer.</td>
    </tr>
  <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>When the output yDtype is int32, the size of the last dimension of y is not 1/8 of the size of the last dimension of x, and the last dimension of x is not divisible by 8.</td>
    </tr>
        <tr>
      <td>gamma is not 1D or 2D, beta (if present) is not 1D or 2D, or scale is not 1D.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_INNER_NULLPTR</td>
      <td rowspan="7">561103</td>
      <td>The number of dimensions and data type of the input or output parameters are out of range.</td>
    </tr>
    <tr>
      <td>The data type combination of input or output parameters is not within the described Constraints.</td>
    </tr>
    <tr>
      <td>The shapes of input x and output y are not exactly the same.</td>
    </tr>
    <tr>
      <td>When output yDtype is int4, the size of the last dimension of x is not an even number.</td>
    </tr>
    <tr>
      <td>The last dimensions of gamma and beta (if present) are different, or their last dimension sizes are not equal to the size of the last dimension of x, or their data types are different.</td>
    </tr>
    <tr>
      <td>The number of dimensions of rstdOut is different from that of x, or the shape of rstdOut is inconsistent with the preceding dimensions of the shape of input x (the preceding dimensions refer to the dimensions of x minus 1), or the last dimension of rstdOut is not 1.</td>
    </tr>
    <tr>
      <td>The shapes of scale and offsetOptional (if present) are not exactly the same, or the axis size is not 1 or the size of the last dimension of x, or the data types do not satisfy the following Constraints.</td>
    </tr>

  </tbody></table>

## aclnnRmsNormQuantV3

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
      <td>Workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device side, obtained by calling the first-phase API aclnnRmsNormQuantV3GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, containing the operator computation flow.</td>
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

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>: When the data type of <code>y</code> is INT4, the last dimension of <code>x</code>, <code>gamma</code>, and <code>beta</code> must be an even number.
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>: When the data type of <code>y</code> is INT32, the last dimension of <code>y</code> must be 1/8 of the last dimension of <code>x</code>.
<!-- end id8 -->
- Supported data types for each product model:

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:

    | X Data Type | Gamma Data Type | Scale Data Type | offsetOptional Data Type | betaOptional Data Type | Epsilon Data Type | y Data Type | Rstd Data Type |
    | --------- | ------------- |  ------------- | -------------- |------------- | --------- |--------- |--------- |
    | FLOAT16   | FLOAT16       |  FLOAT16       | INT8           |FLOAT16       | DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      | FLOAT32       |
    | BFLOAT16   | BFLOAT16       |  BFLOAT16       | INT8           | BFLOAT16       |DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      | FLOAT32       |
    | FLOAT16   | FLOAT16       |  FLOAT16       | FLOAT16           | FLOAT16       |DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      | FLOAT32       |
    | BFLOAT16   | BFLOAT16       |  BFLOAT16       | BFLOAT16           |BFLOAT16       | DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      | FLOAT32       |
    | FLOAT32   | FLOAT32       |  FLOAT32       | FLOAT32           |FLOAT32       | DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      | FLOAT32       |
    | FLOAT16   | FLOAT16       |  FLOAT32       | INT32           |FLOAT16       |DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      | FLOAT32       |
    | BFLOAT16   | BFLOAT16       |  FLOAT32      | INT32           |BFLOAT16       | DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      | FLOAT32       |
    | FLOAT16   | FLOAT16       |  FLOAT32       | FLOAT32           | FLOAT16       |DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8      | FLOAT32       |
    | BFLOAT16   | BFLOAT16       |  FLOAT32       | FLOAT32           | BFLOAT16       |DOUBLE      |INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8     | FLOAT32       |
  <!-- end id9 -->

- Deterministic computation:
  - aclnnRmsNormQuantV3 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="950" id10 -->
```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <memory>
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm_quant_v3.h"

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

// Check whether the current hardware is supported.
bool CheckHardwareSupport()
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        LOG_PRINT("Warning: Cannot get SOC name, skip hardware check\n");
        return true;
    }

    LOG_PRINT("Current SOC: %s\n", socName);

    // This operator supports only Ascend 950PR&950DT products.
    if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
        return true;
    }

    LOG_PRINT("Warning: This operator only supports Ascend 950PR&950DT products, current SOC '%s' is not supported. Skip test.\n",
              socName);
    return false;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
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

    // Call aclrtMemcpy to copy data from the host to the device memory.
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
    // 1. (Boilerplate) device/stream initialization
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Check hardware support.
    if (!CheckHardwareSupport()) {
        LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
        Finalize(deviceId, stream);
        return 0;
    }

    // 2. Construct the inputs and outputs.
    // x: [2, 64] FP16, gamma: [64] FP16, scale: [1] FP16, offset: [1] INT8, beta: [64] FP16
    // y: [2, 64] INT8, rstd: [2, 1] FLOAT32
    std::vector<int64_t> x_shape = {2, 64};
    std::vector<int64_t> gamma_shape = {64};
    std::vector<int64_t> scale_shape = {1};
    std::vector<int64_t> offset_shape = {1};
    std::vector<int64_t> beta_shape = {64};
    std::vector<int64_t> y_shape = {2, 64};
    std::vector<int64_t> rstd_shape = {2, 1};

    void* x_device_addr = nullptr;
    void* gamma_device_addr = nullptr;
    void* scale_device_addr = nullptr;
    void* offset_device_addr = nullptr;
    void* beta_device_addr = nullptr;
    void* y_device_addr = nullptr;
    void* rstd_device_addr = nullptr;
    void* workspace_addr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* y = nullptr;
    aclTensor* rstd = nullptr;

    // Initialize the input data.
    // x: FP16 0.5 = 0x3800
    std::vector<uint16_t> x_host_data(GetShapeSize(x_shape), 0x3800);
    // gamma: FP16 1.5 = 0x3e00
    std::vector<uint16_t> gamma_host_data(GetShapeSize(gamma_shape), 0x3e00);
    // scale: FP16 1.0 = 0x3C00
    std::vector<uint16_t> scale_host_data(GetShapeSize(scale_shape), 0x3C00);
    // offset: INT8 0
    std::vector<int8_t> offset_host_data(GetShapeSize(offset_shape), 0);
    // beta: FP16 0.0 = 0x0000
    std::vector<uint16_t> beta_host_data(GetShapeSize(beta_shape), 0);
    // Initialize the output.
    std::vector<int8_t> y_host_data(GetShapeSize(y_shape), 0);
    std::vector<float> rstd_host_data(GetShapeSize(rstd_shape), 0.0f);

    double epsilon = 1e-6;
    bool divMode = true;
    bool outputRstd = true;

    LOG_PRINT("Input x shape: [2, 64], dtype: FP16\n");
    LOG_PRINT("Gamma shape: [64], dtype: FP16\n");
    LOG_PRINT("Scale shape: [1], dtype: FP16\n");
    LOG_PRINT("Offset shape: [1], dtype: INT8\n");
    LOG_PRINT("Beta shape: [64], dtype: FP16\n");
    LOG_PRINT("Output y shape: [2, 64], dtype: INT8\n");
    LOG_PRINT("Output rstd shape: [2, 1], dtype: FLOAT32\n");
    LOG_PRINT("epsilon: %e, divMode: %s, outputRstd: %s\n",
              epsilon, divMode ? "true" : "false", outputRstd ? "true" : "false");

    // Create the x aclTensor (FP16).
    ret = CreateAclTensor(x_host_data, x_shape, &x_device_addr, aclDataType::ACL_FLOAT16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(x_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor x failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // Create the gamma aclTensor (FP16).
    ret = CreateAclTensor(gamma_host_data, gamma_shape, &gamma_device_addr, aclDataType::ACL_FLOAT16, &gamma);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> gammaTensorPtr(gamma, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> gammaDeviceAddrPtr(gamma_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor gamma failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // Create the scale aclTensor (FP16).
    ret = CreateAclTensor(scale_host_data, scale_shape, &scale_device_addr, aclDataType::ACL_FLOAT16, &scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scaleDeviceAddrPtr(scale_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor scale failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // Create the offset aclTensor (INT8).
    ret = CreateAclTensor(offset_host_data, offset_shape, &offset_device_addr, aclDataType::ACL_INT8, &offset);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> offsetTensorPtr(offset, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> offsetDeviceAddrPtr(offset_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor offset failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // Create the beta aclTensor (FP16).
    ret = CreateAclTensor(beta_host_data, beta_shape, &beta_device_addr, aclDataType::ACL_FLOAT16, &beta);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> betaTensorPtr(beta, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> betaDeviceAddrPtr(beta_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor beta failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // Create the y aclTensor (INT8 output).
    ret = CreateAclTensor(y_host_data, y_shape, &y_device_addr, aclDataType::ACL_INT8, &y);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(y_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor y failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // Create the rstd aclTensor (FLOAT32 output).
    ret = CreateAclTensor(rstd_host_data, rstd_shape, &rstd_device_addr, aclDataType::ACL_FLOAT, &rstd);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> rstdTensorPtr(rstd, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> rstdDeviceAddrPtr(rstd_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor rstd failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 3. Call the CANN operator library API
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    // Call the first-phase API of aclnnRmsNormQuantV3
    LOG_PRINT("Calling aclnnRmsNormQuantV3GetWorkspaceSize...\n");
    ret = aclnnRmsNormQuantV3GetWorkspaceSize(
        x, gamma, scale, offset, beta, epsilon, divMode, outputRstd,
        y, rstd, &workspace_size, &executor);

    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRmsNormQuantV3GetWorkspaceSize failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    LOG_PRINT("Workspace size: %lu bytes (%.2f KB)\n", workspace_size, workspace_size / 1024.0);

    // Allocate device memory based on the workspaceSize computed by the first-phase API
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);
        workspaceAddrPtr.reset(workspace_addr);
    }

    // Call the second-phase API of aclnnRmsNormQuantV3
    LOG_PRINT("Calling aclnnRmsNormQuantV3...\n");
    ret = aclnnRmsNormQuantV3(workspaceAddrPtr.get(), workspace_size, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRmsNormQuantV3 failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 4. Wait synchronously for task execution to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 5. Obtain the output values and copy the results from the device side to the host side
    {
        // Copy the y result (INT8)
        auto size = GetShapeSize(y_shape);
        std::vector<int8_t> y_result(size, 0);
        ret = aclrtMemcpy(y_result.data(), y_result.size() * sizeof(y_result[0]),
                          yDeviceAddrPtr.get(), size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);

        LOG_PRINT("Output y (first 10 values):\n");
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            LOG_PRINT("  y[%ld] = %d\n", i, y_result[i]);
        }

        // Copy the rstd result (FLOAT32)
        size = GetShapeSize(rstd_shape);
        std::vector<float> rstd_result(size, 0.0f);
        ret = aclrtMemcpy(rstd_result.data(), rstd_result.size() * sizeof(rstd_result[0]),
                          rstdDeviceAddrPtr.get(), size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("copy rstd from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);

        LOG_PRINT("Output rstd values:\n");
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("  rstd[%ld] = %f\n", i, rstd_result[i]);
        }
    }

    LOG_PRINT("\n=== RmsNormQuantV3 Test PASSED ===\n");

    // 6. Resources are automatically released
    Finalize(deviceId, stream);

    return 0;
}
```
<!-- end id10 -->
