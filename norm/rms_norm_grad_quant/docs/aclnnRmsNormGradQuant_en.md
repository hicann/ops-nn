# aclnnRmsNormGradQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:35:15.931Z -->

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

- Description: RmsNormGrad computes the gradient of RmsNorm, that is, the gradient of the input tensor during backward propagation. RmsNormGradQuant fuses the RmsNormGrad and Quantize operators: quant is computed after RmsNormGrad computes dx, reducing move-in and move-out operations.
- Operator formula:

  $$
  dx_i= (dy_i * g_i - \frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * \operatorname{Mean}(\mathbf{y})) * \frac{1} {\operatorname{Rms}(\mathbf{x})},  \quad \text { where } \operatorname{Mean}(\mathbf{y}) = \frac{1}{n}\sum_{i=1}^n (dy_i * g_i * x_i * \frac{1}{\operatorname{Rms}(\mathbf{x})})
  $$

  - When divMode is true:

    $$
    dx_i\_quant=round((dx_i/scales\_x)+offset\_x)
    $$

  - When divMode is false:

    $$
    dx_i\_quant=round((dx_i*scales\_x)+offset\_x)
    $$

    $$
    dg_i = \frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * dy_i
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call `aclnnRmsNormGradQuantGetWorkspaceSize` first to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, call `aclnnRmsNormGradQuant` to perform the computation.

```Cpp
aclnnStatus aclnnRmsNormGradQuantGetWorkspaceSize(
  const aclTensor *dy,
  const aclTensor *x,
  const aclTensor *rstd,
  const aclTensor *gamma,
  const aclTensor *scalesX,
  const aclTensor *offsetXOptional,
  const char      *quantMode,
  bool             divMode,
  aclTensor       *dxOut,
  aclTensor       *dgammaOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnRmsNormGradQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnRmsNormGradQuantGetWorkspaceSize

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
      <td>dy (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the gradient propagated back in backward propagation, corresponding to dy in the formula.</td>
      <td><ul><li>Supports an empty tensor.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input of the forward operator and the data being normalized, corresponding to x in the formula.</td>
      <td><ul><li>Supports empty Tensor.</li><li>Shape and data type remain consistent with dy.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the intermediate computation result of the forward operator, corresponding to Rms(x) in the formula.</td>
      <td><ul><li>Supports empty Tensor.</li><li>The shape must satisfy rstd_shape = x_shape[0:n], where n < x_shape.dims() and n remains consistent with the n of gamma.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the scaling factor (weight) used by the forward operator for normalization computation, corresponding to gamma in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape must satisfy gamma_shape = x_shape[n:], where n < x_shape.dims().</li><li>The dtype must be the same as dy or FLOAT32.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scalesX (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the quantization scale factor of the input gradient, which is scales_x in the corresponding formula.</td>
      <td><ul><li>Supports empty Tensor.</li><li>The shape is [1] and the dimension is 1.</li><li>dtype is the same as dy or FLOAT32.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>offsetXOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input gradient quantization zero point, corresponding to offset_x in the formula.</td>
      <td><ul><li>Supports empty Tensor.</li><li>The shape is [1] and the dimension is 1.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>quantMode (char*)</td>
      <td>Input</td>
      <td>Quantization mode.</td>
      <td><ul><li>Only "static" is supported, indicating the static quantization mode.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divMode (bool)</td>
      <td>Input</td>
      <td>Parameter that determines whether the quantization formula uses division. The data type is bool.</td>
      <td><ul><li>True and False are supported.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dxOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the gradient of input x, corresponding to dx in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape remains consistent with the shape of dy.</li></ul></td>
      <td>HIFLOAT8, INT8</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dgammaOut(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the gradient of gamma, corresponding to dg in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape must remain consistent with the shape of the input parameter gamma.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize(uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied by the user on the Device.</td>
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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>If a required input, output, or required attribute is passed in as a null pointer, 161001 is returned.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of the input or output is not within the supported range.</td>
    </tr>
  </tbody></table>

## aclnnRmsNormGradQuant

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
      <td>Workspace memory address applied by the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied by the Device, obtained by the first-phase API aclnnRmsNormGradQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation process.</td>
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

- Supported data types for each product:

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:

    | `dy` Data Type | `x` Data Type | `rstd` Data Type | `gamma` Data Type | `scalesX` Data Type | `offsetXOptional` Data Type | `dxOut` Data Type | `dgammaOut` Data Type |
    | ------------ | ----------- | -------------- | --------------- | ------------------------- | ---------------------------- | --------------- | ------------------- |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT32         | FLOAT32                   | INT32                        | HIFLOAT8        | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | FLOAT32         | FLOAT32                   | INT32                        | HIFLOAT8        | FLOAT32             |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT16         | FLOAT32                   | INT32                        | HIFLOAT8        | FLOAT32             |
    | FLOAT32      | FLOAT32     | FLOAT32        | FLOAT32         | FLOAT32                   | INT32                        | HIFLOAT8        | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | BFLOAT16        | FLOAT32                   | INT32                        | HIFLOAT8        | FLOAT32             |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT16         | FLOAT16                   | INT32                        | HIFLOAT8        | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | BFLOAT16        | BFLOAT16                  | INT32                        | HIFLOAT8        | FLOAT32             |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT32         | FLOAT16                   | INT32                        | HIFLOAT8        | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | FLOAT32         | BFLOAT16                  | INT32                        | HIFLOAT8        | FLOAT32             |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT32         | FLOAT32                   | INT32                        | INT8            | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | FLOAT32         | FLOAT32                   | INT32                        | INT8            | FLOAT32             |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT16         | FLOAT32                   | INT32                        | INT8            | FLOAT32             |
    | FLOAT32      | FLOAT32     | FLOAT32        | FLOAT32         | FLOAT32                   | INT32                        | INT8            | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | BFLOAT16        | FLOAT32                   | INT32                        | INT8            | FLOAT32             |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT16         | FLOAT16                   | INT32                        | INT8            | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | BFLOAT16        | BFLOAT16                  | INT32                        | INT8            | FLOAT32             |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT32         | FLOAT16                   | INT32                        | INT8            | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | FLOAT32         | BFLOAT16                  | INT32                        | INT8            | FLOAT32             |
  <!-- end id7 -->

- Deterministic computation:
  - aclnnRmsNormGradQuant defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <memory>
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm_grad_quant.h"

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
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. Initialize the device/stream.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output.
    // RmsNormGradQuant: dy[M, D], x[M, D], rstd[M], gamma[D], scalesX[1], offsetX[1] -> dx[M, D], dgamma[D]
    std::vector<int64_t> dy_shape = {8, 64};
    std::vector<int64_t> x_shape = {8, 64};
    std::vector<int64_t> rstd_shape = {8};
    std::vector<int64_t> gamma_shape = {64};
    std::vector<int64_t> scales_x_shape = {1};
    std::vector<int64_t> offset_x_shape = {1};
    std::vector<int64_t> dx_shape = {8, 64};
    std::vector<int64_t> dgamma_shape = {64};

    void* dy_device_addr = nullptr;
    void* x_device_addr = nullptr;
    void* rstd_device_addr = nullptr;
    void* gamma_device_addr = nullptr;
    void* scales_x_device_addr = nullptr;
    void* offset_x_device_addr = nullptr;
    void* dx_device_addr = nullptr;
    void* dgamma_device_addr = nullptr;
    void* workspace_addr = nullptr;

    aclTensor* dy = nullptr;
    aclTensor* x = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* scalesX = nullptr;
    aclTensor* offsetX = nullptr;
    aclTensor* dxOut = nullptr;
    aclTensor* dgammaOut = nullptr;

    // Initialize the input data.
    // dy, x: FP16, 1.0 = 0x3C00
    std::vector<uint16_t> dy_host_data(GetShapeSize(dy_shape), 0x3C00);
    std::vector<uint16_t> x_host_data(GetShapeSize(x_shape), 0x3C00);
    // rstd: FP32, 1.0
    std::vector<float> rstd_host_data(GetShapeSize(rstd_shape), 1.0f);
    // gamma: FP32, 1.0
    std::vector<float> gamma_host_data(GetShapeSize(gamma_shape), 1.0f);
    // scalesX: FP32, 1.0
    std::vector<float> scales_x_host_data(GetShapeSize(scales_x_shape), 1.0f);
    // offsetX: INT32, 0
    std::vector<int32_t> offset_x_host_data(GetShapeSize(offset_x_shape), 0);
    // dx: INT8 output
    std::vector<int8_t> dx_host_data(GetShapeSize(dx_shape), 0);
    // dgamma: FP32 output
    std::vector<float> dgamma_host_data(GetShapeSize(dgamma_shape), 0.0f);

    char quantMode[] = "static";
    bool divMode = true;

    LOG_PRINT("Input shape: dy[8, 64], x[8, 64], rstd[8], gamma[64]\n");
    LOG_PRINT("Output shape: dx[8, 64](INT8), dgamma[64](FP32)\n");
    LOG_PRINT("quantMode: static, divMode: true\n");

    // Create the dy aclTensor (FP16).
    ret = CreateAclTensor(dy_host_data, dy_shape, &dy_device_addr, aclDataType::ACL_FLOAT16, &dy);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> dyPtr(dy, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> dyDevPtr(dy_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor dy failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // Create the x aclTensor (FP16).
    ret = CreateAclTensor(x_host_data, x_shape, &x_device_addr, aclDataType::ACL_FLOAT16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDevPtr(x_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor x failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // Create the rstd aclTensor (FP32).
    ret = CreateAclTensor(rstd_host_data, rstd_shape, &rstd_device_addr, aclDataType::ACL_FLOAT, &rstd);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> rstdPtr(rstd, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> rstdDevPtr(rstd_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor rstd failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // Create gamma aclTensor (FP32).
    ret = CreateAclTensor(gamma_host_data, gamma_shape, &gamma_device_addr, aclDataType::ACL_FLOAT, &gamma);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> gammaPtr(gamma, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> gammaDevPtr(gamma_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor gamma failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // Create scalesX aclTensor (FP32).
    ret = CreateAclTensor(scales_x_host_data, scales_x_shape, &scales_x_device_addr, aclDataType::ACL_FLOAT, &scalesX);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scalesXPtr(scalesX, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scalesXDevPtr(scales_x_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor scalesX failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // Create offsetX aclTensor (INT32).
    ret = CreateAclTensor(offset_x_host_data, offset_x_shape, &offset_x_device_addr, aclDataType::ACL_INT32, &offsetX);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> offsetXPtr(offsetX, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> offsetXDevPtr(offset_x_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor offsetX failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // Create dxOut aclTensor (INT8).
    ret = CreateAclTensor(dx_host_data, dx_shape, &dx_device_addr, aclDataType::ACL_INT8, &dxOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> dxOutPtr(dxOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> dxDevPtr(dx_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor dxOut failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // Create dgammaOut aclTensor (FP32).
    ret = CreateAclTensor(dgamma_host_data, dgamma_shape, &dgamma_device_addr, aclDataType::ACL_FLOAT, &dgammaOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> dgammaOutPtr(dgammaOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> dgammaDevPtr(dgamma_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor dgammaOut failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 3. Call the CANN operator library APIs.
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    // Call the first-phase API of aclnnRmsNormGradQuant.
    LOG_PRINT("Calling aclnnRmsNormGradQuantGetWorkspaceSize...\n");
    ret = aclnnRmsNormGradQuantGetWorkspaceSize(
        dy, x, rstd, gamma, scalesX, offsetX,
        quantMode, divMode,
        dxOut, dgammaOut, &workspace_size, &executor);

    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRmsNormGradQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    LOG_PRINT("Workspace size: %lu bytes (%.2f KB)\n", workspace_size, workspace_size / 1024.0);

    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);
        workspaceAddrPtr.reset(workspace_addr);
    }

    // Call the second-phase API of aclnnRmsNormGradQuant.
    LOG_PRINT("Calling aclnnRmsNormGradQuant...\n");
    ret = aclnnRmsNormGradQuant(workspaceAddrPtr.get(), workspace_size, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRmsNormGradQuant failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 4. Synchronize and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 5. Obtain the output values, and copy the results on the device side to the host side.
    {
        // Copy the dx result (INT8).
        auto size = GetShapeSize(dx_shape);
        std::vector<int8_t> dx_result(size, 0);
        ret = aclrtMemcpy(dx_result.data(), dx_result.size() * sizeof(dx_result[0]),
                          dxDevPtr.get(), size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("copy dx from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);

        LOG_PRINT("Output dx (first 10 values):\n");
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            LOG_PRINT("  dx[%ld] = %d\n", i, dx_result[i]);
        }

        // Copy the dgamma result (FP32).
        size = GetShapeSize(dgamma_shape);
        std::vector<float> dgamma_result(size, 0.0f);
        ret = aclrtMemcpy(dgamma_result.data(), dgamma_result.size() * sizeof(dgamma_result[0]),
                          dgammaDevPtr.get(), size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("copy dgamma from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);

        LOG_PRINT("Output dgamma (first 10 values):\n");
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            LOG_PRINT("  dgamma[%ld] = %f\n", i, dgamma_result[i]);
        }
    }

    LOG_PRINT("\n=== RmsNormGradQuant Test PASSED ===\n");

    // 6. Resources are automatically released.
    Finalize(deviceId, stream);

    return 0;
}
```
