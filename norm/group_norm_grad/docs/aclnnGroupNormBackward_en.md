# aclnnGroupNormBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:04:45.357Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Backward computation of [aclnnGroupNorm](../../group_norm/docs/aclnnGroupNorm.md). Computes the gradients of the input tensor to update model parameters during backward propagation.
- Formula:

  $$
  \hat{x} = (x - mean) \cdot rstd
  $$

  $$
  gradBetaOut = \sum_{i=1}^n gradOut
  $$

  $$
  gradGammaOut = \sum_{i=1}^n (gradOut \cdot \hat{x})
  $$

  $$
  gradInput = rstd \cdot gamma \begin{bmatrix}
  gradOut - \frac{1}{N}  (gradBetaOut + \hat{x} \cdot gradGammaOut)
  \end{bmatrix}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnGroupNormBackwardGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the computation process of the operator, and then call "aclnnGroupNormBackward" to perform the computation.

```Cpp
aclnnStatus aclnnGroupNormBackwardGetWorkspaceSize(
  const aclTensor*     gradOut,
  const aclTensor*     input,
  const aclTensor*     mean,
  const aclTensor*     rstd,
  const aclTensor*     gamma,
  int64_t              N,
  int64_t              C,
  int64_t              HxW,
  int64_t              group,
  const aclBoolArray*  outputMask,
  aclTensor*           gradInput,
  aclTensor*           gradGammaOut,
  aclTensor*           gradBetaOut,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnGroupNormBackward(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnGroupNormBackwardGetWorkspaceSize

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
      <td>gradOut (aclTensor*)</td>
      <td>Input</td>
      <td>Gradient tensor for the backward computation, corresponding to `gradOut` in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>The data type is the same as that of `input`.</li><li>The number of elements must be equal to N*C*HxW.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input (aclTensor*)</td>
      <td>Input</td>
      <td>First input of the forward computation, corresponding to `x` in the formula.</td>
      <td><ul><li>Supports null tensor.</li><li>The data type is the same as `gradOut`.</li><li>The number of elements must equal N*C*HxW.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean(aclTensor*)</td>
      <td>Input</td>
      <td>Second output of the forward computation, representing the mean value of each group after input is grouped. It corresponds to `mean` in the formula.</td>
      <td><ul><li>Supports null tensor.</li><li>The number of elements must equal N*group.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd (aclTensor*)</td>
      <td>Input</td>
      <td>The third output of forward computation. It represents the reciprocal of the standard deviation of each group after the input is grouped and corresponds to `rstd` in the formula.</td>
      <td><ul><li>Supports null tensor.</li><li>Has the same data type as `mean`.</li><li>The number of elements must be equal to N*group.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Scaling coefficient for each channel, corresponding to `gamma` in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>Optional input: a null pointer is allowed; when it is null, γ is set to 1 in the calculation.</li><li>The data type must be the same as that of `mean`.</li><li>The number of elements must be the same as that of `C`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>N (int64_t)</td>
      <td>Input</td>
      <td>Indicates the space size of input `gradOut` in the N dimension.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>C (int64_t)</td>
      <td>Input</td>
      <td>Indicates the space size of input `gradOut` in the C dimension.</td>
      <td><ul><li>The value of C is less than or equal to 240000.</li><li>C must be divisible by group, and C/group ≤ 8000.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>HxW (int64_t)</td>
      <td>Input</td>
      <td>Represents the space size of input `gradOut` excluding the N and C dimensions.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group (int64_t)</td>
      <td>Input</td>
      <td>Indicates that the C dimension of the input `gradOut` is divided into group groups.</td>
      <td><ul><li>The value of group must be greater than 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMask(aclBoolArray*)</td>
      <td>Input</td>
      <td>Specifies the output mask, indicating whether to output `gradInput`, `gradGammaOut`, and `gradBetaOut`.</td>
      <td><ul><li>The size is 3. The three elements indicate whether to output `gradInput`, `gradGammaOut`, and `gradBetaOut` respectively. If the value is true, the corresponding output is generated; otherwise, null is returned at the corresponding position.</li></ul></td>
      <td>BoolArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput(aclTensor*)</td>
      <td>Output</td>
      <td>Gradient of the computed output, used to update the gradient of the input data. Corresponds to `gradInput` in the corresponding formula.</td>
      <td><ul><li>Supports null tensor.</li><li>Data type is the same as `gradOut`.</li><li>Shape is the same as `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradGammaOut(aclTensor*)</td>
      <td>Output</td>
      <td>Gradient of the computed output, used to update the gradient of the scale parameter. Corresponds to `gradGammaOut` in the corresponding formula.</td>
      <td><ul><li>Support null tensor.</li><li>Same data type as `mean`.</li><li>Same shape as `gamma`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBetaOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the gradient of the computed output, used to update the gradient of the bias parameter. Corresponds to `gradBetaOut` in the corresponding formula.</td>
      <td><ul><li>Support null tensor.</li><li>Same data type as `mean`.</li><li>Same shape as `gamma`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
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
      <td>Returns the op executor that contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910" id7 -->
  - <term>Atlas training products</term>:

    - The `gradOut`, `input`, `mean`, `rstd`, `gamma`, `gradInput`, `gradGammaOut`, and `gradBetaOut` parameters do not support the BFLOAT16 data type.
    - The data types of the `mean` and `gradOut` parameters are the same.
  <!-- end id7 -->

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    The data types of the `mean` and `gradOut` parameters are the same.
  <!-- end id8 -->

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:

    The supported data type correspondence between `mean` and `gradOut` is as follows:

    | Input Parameters | gradOut | Mean |
    |--|--|--|
    |Data type | FLOAT32 | FLOAT32 |
    |Data type | FLOAT16 | FLOAT16 |
    |Data type | FLOAT16 | FLOAT32 |
    |Data type | BFLOAT16 | BFLOAT16 |
    |Data type | BFLOAT16 | FLOAT32 |
  <!-- end id9 -->

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
      <td rowspan="4">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="4">161001</td>
      <td>The passed-in gradOut, input, mean, or rstd is a null pointer.</td>
    </tr>
    <tr>
      <td>The passed-in gradInput is a null pointer when outputMask[0] is true.</td>
    </tr>
    <tr>
      <td>outputMask[1] is true but the passed-in gradGammaOut is a null pointer.</td>
    </tr>
    <tr>
      <td>outputMask[2] is true but the passed-in gradBetaOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>The data type of gradOut is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data types of input, mean, gamma, rstd, and gradOut do not comply with the constraints in Parameters.</td>
    </tr>
    <tr>
      <td>The length of outputMask is not 3.</td>
    </tr>
    <tr>
      <td>outputMask[0] is true but the shape of gradInput is different from that of input.</td>
    </tr>
    <tr>
      <td>outputMask[1] is true but the shape of gradGammaOut is different from that of gamma.</td>
    </tr>
    <tr>
      <td>outputMask[2] is true but the shape of gradBetaOut is different from that of gamma.</td>
    </tr>
    <tr>
      <td>group is not greater than 0, or C is greater than 240000.</td>
    </tr>
    <tr>
      <td>C is not divisible by group, or C/group is greater than 8000.</td>
    </tr>
    <tr>
      <td>The number of elements in input is not equal to N*C*HxW.</td>
    </tr>
    <tr>
      <td>The number of elements in mean is not equal to N*group.</td>
    </tr>
    <tr>
      <td>The number of elements in rstd is not equal to N*group.</td>
    </tr>
    <tr>
      <td>gamma is not a null pointer and the number of elements in gamma is not C.</td>
    </tr>
  </tbody></table>

## aclnnGroupNormBackward

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by the first-phase API aclnnGroupNormBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="A3,910b,910" id10 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas training products</term>: aclnnGroupNormBackward defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtCtxSetSysParamOpt.
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnGroupNormBackward defaults to a deterministic implementation.
  <!-- end id11 -->

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_group_norm_backward.h"

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
    // Boilerplate for resource initialization
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
    // Call aclrtMalloc to apply for device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to device-side memory.
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
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. The construction must be customized based on the API.
    std::vector<int64_t> gradOutShape = {2, 3, 4};
    std::vector<int64_t> inputShape = {2, 3, 4};
    std::vector<int64_t> meanShape = {2, 1};
    std::vector<int64_t> rstdShape = {2, 1};
    std::vector<int64_t> gammaShape = {3};
    std::vector<int64_t> gradInputShape = {2, 3, 4};
    std::vector<int64_t> gradGammaOutShape = {3};
    std::vector<int64_t> gradBetaOutShape = {3};
    void* gradOutDeviceAddr = nullptr;
    void* inputDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    void* gradGammaOutDeviceAddr = nullptr;
    void* gradBetaOutDeviceAddr = nullptr;
    aclTensor* gradOut = nullptr;
    aclTensor* input = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* gradInput = nullptr;
    aclTensor* gradGammaOut = nullptr;
    aclTensor* gradBetaOut = nullptr;
    std::vector<float> gradOutHostData = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    std::vector<float> inputHostData = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                                        13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    std::vector<float> meanHostData = {6.5, 18.5};
    std::vector<float> rstdHostData = {0.2896827, 0.2896827};
    std::vector<float> gammaHostData = {1.0, 1.0, 1.0};
    std::vector<float> gradInputHostData = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> gradGammaOutHostData = {0.0, 0.0, 0.0};
    std::vector<float> gradBetaOutHostData = {0.0, 0.0, 0.0};
    int64_t N = 2;
    int64_t C = 3;
    int64_t HxW = 4;
    int64_t group = 1;
    std::array<bool, 3> outputMaskData = {true, true, true};
    // Create the gradOut aclTensor.
    ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the input aclTensor.
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the mean aclTensor.
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the rstd aclTensor.
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gamma aclTensor.
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    auto outputMask = aclCreateBoolArray(outputMaskData.data(), outputMaskData.size());
    CHECK_RET(outputMask != nullptr, return ACL_ERROR_INTERNAL_ERROR);
    // Create the gradInput aclTensor.
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gradGammaOut aclTensor.
    ret = CreateAclTensor(
        gradGammaOutHostData, gradGammaOutShape, &gradGammaOutDeviceAddr, aclDataType::ACL_FLOAT, &gradGammaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gradBetaOut aclTensor.
    ret = CreateAclTensor(
        gradBetaOutHostData, gradBetaOutShape, &gradBetaOutDeviceAddr, aclDataType::ACL_FLOAT, &gradBetaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the specific HostApi.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnGroupNormBackward.
    ret = aclnnGroupNormBackwardGetWorkspaceSize(
        gradOut, input, mean, rstd, gamma, N, C, HxW, group, outputMask, gradInput, gradGammaOut, gradBetaOut,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGroupNormBackward.
    ret = aclnnGroupNormBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values, and copy the results on device memory to host memory. Modify this step based on the specific API definition.
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> gradInputResultData(size, 0);
    ret = aclrtMemcpy(
        gradInputResultData.data(), gradInputResultData.size() * sizeof(gradInputResultData[0]), gradInputDeviceAddr,
        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradInputResultData[%ld] is: %f\n", i, gradInputResultData[i]);
    }

    size = GetShapeSize(gradGammaOutShape);
    std::vector<float> gradGammaOutResultData(size, 0);
    ret = aclrtMemcpy(
        gradGammaOutResultData.data(), gradGammaOutResultData.size() * sizeof(gradGammaOutResultData[0]),
        gradGammaOutDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradGammaOutResultData[%ld] is: %f\n", i, gradGammaOutResultData[i]);
    }

    size = GetShapeSize(gradBetaOutShape);
    std::vector<float> gradBetaOutResultData(size, 0);
    ret = aclrtMemcpy(
        gradBetaOutResultData.data(), gradBetaOutResultData.size() * sizeof(gradBetaOutResultData[0]),
        gradBetaOutDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradBetaOutResultData[%ld] is: %f\n", i, gradBetaOutResultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify this step based on the specific API definition.
    aclDestroyTensor(gradOut);
    aclDestroyTensor(input);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);
    aclDestroyTensor(gamma);
    aclDestroyTensor(gradInput);
    aclDestroyTensor(gradGammaOut);
    aclDestroyTensor(gradBetaOut);

    // 7. Release device resources. Modify this step based on the specific API definition.
    aclrtFree(gradOutDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    aclrtFree(gradGammaOutDeviceAddr);
    aclrtFree(gradBetaOutDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
