# aclnnRmsNormGrad

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/rms_norm_grad)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     √    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description: Performs backpropagation of [aclnnRmsNorm](../../rms_norm/docs/aclnnRmsNorm_en.md). It is used to compute the gradient of **RmsNorm**, that is, compute the gradient of the input tensor during backpropagation.
- Formula:

  - Forward propagation:

  $$
  \operatorname{RmsNorm}(x_i)=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  - Backward propagation:

  $$
  dx_i= (dy_i * g_i - \frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * \operatorname{Mean}(\mathbf{y})) * \frac{1} {\operatorname{Rms}(\mathbf{x})},  \quad \text { where } \operatorname{Mean}(\mathbf{y}) = \frac{1}{n}\sum_{i=1}^n (dy_i * g_i * x_i * \frac{1}{\operatorname{Rms}(\mathbf{x})})
  $$

  $$
  dg_i = \frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * dy_i
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnRmsNormGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnRmsNormGrad` is called to perform computation.

```Cpp
aclnnStatus aclnnRmsNormGradGetWorkspaceSize(
  const aclTensor *dy,
  const aclTensor *x,
  const aclTensor *rstd,
  const aclTensor *gamma,
  const aclTensor *dxOut,
  const aclTensor *dgammaOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnRmsNormGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnRmsNormGradGetWorkspaceSize

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
      <td>dy</td>
      <td>Input</td>
      <td>Backpropagated gradient, corresponding to `dy` in the formula.</td>
      <td>Empty tensors are supported.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input of the forward operator, indicating the normalized data and corresponding to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as that of the input parameter `dy`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Input</td>
      <td>Intermediate computation result of the forward operator, corresponding to `Rms(x)` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must meet the requirement: rstd_shape = x_shape[0:n], n < x_shape.dims(), where **n** is the same as that of `gamma`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Scaling factor (weight) for normalization computation of the forward operator, corresponding to `gamma` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must meet the requirement: gamma_shape = x_shape[n:], n < x_shape.dims().</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dxOut</td>
      <td>Output</td>
      <td>Gradient of the input `x`, corresponding to `dx` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as that of the input parameter `dy`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dgammaOut</td>
      <td>Output</td>
      <td>Gradient of `gamma`, corresponding to `dg` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as that of the input parameter `gamma`.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
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

  - <term>Atlas inference series products</term>: The data types of `dy`, `x`, `gamma`, and `dxOut` cannot be BFLOAT16.
  
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
      <td>The passed parameter is a mandatory input, output, or attribute, but is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>The input or output data type is not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The parameter does not meet the requirements in the parameter description.</td>
    </tr>
  </tbody></table>

## aclnnRmsNormGrad

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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnRmsNormGradGetWorkspaceSize.</td>
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

- <term>Atlas inference series products</term>: The length of the last axis of the inputs `x`, `dy`, and `gamma` must be greater than or equal to 32 bytes.

- Description of data types supported by different products:
  - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:
    | `dy`| `x`| `rstd`| `gamma`| `dxOut`| `dgammaOut`|
    | -------- | -------- | -------- | -------- | -------- | -------- |
    | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT16  | FLOAT32  |
    | BFLOAT16 | BFLOAT16 | FLOAT32  | FLOAT32  | BFLOAT16 | FLOAT32  |
    | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT16  | FLOAT16  | FLOAT32  |
    | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | BFLOAT16 | BFLOAT16 | FLOAT32  | BFLOAT16 | BFLOAT16 | FLOAT32  |
  - <term>Atlas inference series products</term>:
    | `dy`| `x`| `rstd`| `gamma`| `dxOut`| `dgammaOut`|
    | -------- | -------- | -------- | -------- | -------- | -------- |
    | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT16  | FLOAT16  | FLOAT32  |
    | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
- Deterministic compute:
  - **aclnnRmsNormGrad** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

## Example


The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm_grad.h"
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
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW, shape.data(),
        shape.size(), *deviceAddr);
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
    // 2. Construct the input and output based on the API definition.
    std::vector<int64_t> gradInputShape = {2, 1, 16};
    std::vector<int64_t> xInputShape = {2, 1, 16};
    std::vector<int64_t> rstdInputShape = {2};
    std::vector<int64_t> gammaInputShape = {16};
    std::vector<int64_t> dxOutputShape = {2, 1, 16};
    std::vector<int64_t> dgammaOutputShape = {16};

    void* gradInputDeviceAddr = nullptr;
    void* xInputDeviceAddr = nullptr;
    void* rstdInputDeviceAddr = nullptr;
    void* gammaInputDeviceAddr = nullptr;
    void* dxOutDeviceAddr = nullptr;
    void* dgammaOutDeviceAddr = nullptr;

    aclTensor* gradInput = nullptr;
    aclTensor* xInput = nullptr;
    aclTensor* rstdInput = nullptr;
    aclTensor* gammaInput = nullptr;
    aclTensor* dxOut = nullptr;
    aclTensor* dgammaOut = nullptr;

    std::vector<float> gradInputHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    std::vector<float> xInputHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    std::vector<float> rstdInputHostData = {1, 2};
    std::vector<float> gammaInputHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    std::vector<float> dxOutHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    std::vector<float> dgammaOutHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    std::vector<int64_t> output1SizeData = {2, 1, 16};
    std::vector<int64_t> output2SizeData = {16};
    std::vector<int64_t> input1SizeData = {2, 1, 16};
    std::vector<int64_t> input2SizeData = {2};
    std::vector<int64_t> input3SizeData = {16};

    ret = CreateAclTensor(gradInputHostData, input1SizeData, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(xInputHostData, input1SizeData, &xInputDeviceAddr, aclDataType::ACL_FLOAT, &xInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(rstdInputHostData, input2SizeData, &rstdInputDeviceAddr, aclDataType::ACL_FLOAT, &rstdInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret =
        CreateAclTensor(gammaInputHostData, input3SizeData, &gammaInputDeviceAddr, aclDataType::ACL_FLOAT, &gammaInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(dxOutHostData, output1SizeData, &dxOutDeviceAddr, aclDataType::ACL_FLOAT, &dxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(dgammaOutHostData, output2SizeData, &dgammaOutDeviceAddr, aclDataType::ACL_FLOAT, &dgammaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnRmsNormGrad.
    ret = aclnnRmsNormGradGetWorkspaceSize(
        gradInput, xInput, rstdInput, gammaInput, dxOut, dgammaOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnRmsNormGrad.
    ret = aclnnRmsNormGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormGrad failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
    auto size_dx = GetShapeSize(gradInputShape);
    std::vector<float> resultData1(size_dx, 0);
    ret = aclrtMemcpy(
        resultData1.data(), resultData1.size() * sizeof(resultData1[0]), dxOutDeviceAddr, size_dx * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size_dx; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData1[i]);
    }
    auto size_dgamma = GetShapeSize(gammaInputShape);
    std::vector<float> resultData2(size_dgamma, 1);
    ret = aclrtMemcpy(
        resultData2.data(), resultData2.size() * sizeof(resultData2[0]), dgammaOutDeviceAddr,
        size_dgamma * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size_dgamma; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData2[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(gradInput);
    aclDestroyTensor(xInput);
    aclDestroyTensor(rstdInput);
    aclDestroyTensor(gammaInput);
    aclDestroyTensor(dxOut);
    aclDestroyTensor(dgammaOut);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(gradInputDeviceAddr);
    aclrtFree(xInputDeviceAddr);
    aclrtFree(rstdInputDeviceAddr);
    aclrtFree(gammaInputDeviceAddr);
    aclrtFree(dxOutDeviceAddr);
    aclrtFree(dgammaOutDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
