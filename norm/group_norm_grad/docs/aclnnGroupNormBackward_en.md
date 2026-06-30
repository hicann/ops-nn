# aclnnGroupNormBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/group_norm_grad)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     √    |

## Function

- Description: Performs backpropagation of [aclnnGroupNorm](../../group_norm/docs/aclnnGroupNorm_en.md). It is used to compute the gradient of the input tensor, so that the model parameters can be updated during backpropagation.
- Formula:
  
  $$
  gradBetaOut = \sum_{i=1}^n gradOut
  $$

  $$
  gradGammaOut = \sum_{i=1}^n (gradOut \cdot \hat{x})
  $$
  
  $$
  gradInput = mean \cdot rstd \cdot gamma \begin{bmatrix}
  gradOut - \frac{1}{N}  (gradBetaOut + \hat{x} \cdot gradGammaOut)
  \end{bmatrix}
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnGroupNormBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnGroupNormBackward** is called to perform computation.

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
      <th>Precaution</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOut</td>
      <td>Input</td>
      <td>Gradient tensor for backpropagation, corresponding to `gradOut` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type is the same as that of `input`. </li><li>The number of elements must be equal to N × C × HxW.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0–8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input</td>
      <td>Input</td>
      <td>First input of forward propagation, corresponding to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type is the same as that of `gradOut`. </li><li>The number of elements must be equal to N × C × HxW.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0–8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Second output of forward propagation, indicating the mean value of each group after input grouping. It corresponds to `mean` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The number of elements must be equal to N × group.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0–8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Input</td>
      <td>Third output of forward propagation, indicating the reciprocal of the standard deviation of each group after input grouping. It corresponds to `rstd` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type is the same as that of `mean`. </li><li>The number of elements must be equal to N × group.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0–8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Scaling coefficient of each channel, corresponding to `gamma` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type is the same as that of <idp:inline displayname="code" id="code1670673215226">mean</idp:inline>. </li><li>The number of elements must be equal to `C`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0–8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>N</td>
      <td>Input</td>
      <td>Space size of the input `gradOut` in the N dimension.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>C</td>
      <td>Input</td>
      <td>Space size of the input `gradOut` in the C dimension.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>HxW</td>
      <td>Input</td>
      <td>Space size of the input `gradOut` in dimensions other than N and C.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group</td>
      <td>Input</td>
      <td>Number of groups into which the C dimension of the input `gradOut` is divided.</td>
      <td><ul><li>The value of group must be greater than 0, and the size of C must be exactly divided by group.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMask</td>
      <td>Input</td>
      <td>Output mask, indicating whether to output `gradInput`, `gradGammaOut`, and `gradBetaOut`.</td>
      <td><ul><li>The size is 3. This parameter indicates whether to output `gradInput`, `gradGammaOut`, and `gradBetaOut`. If the value is true, the output is generated. Otherwise, an empty value is returned at the corresponding position.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>Output</td>
      <td>Computation output gradient for updating input data, corresponding to `gradInput` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type is the same as that of <idp:inline displayname="code" id="code113650620138">gradOut</idp:inline>. </li><li>The shape is the same as that of `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0–8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradGammaOut</td>
      <td>Output</td>
      <td>Computation output gradient for updating the scaling parameter, corresponding to `gradGammaOut` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type is the same as that of <idp:inline displayname="code" id="code137071324229">mean</idp:inline>. </li><li>The shape is the same as that of `gamma`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0–8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBetaOut</td>
      <td>Output</td>
      <td>Computation output gradient for updating the bias parameter, corresponding to `gradBetaOut` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type is the same as that of <idp:inline displayname="code" id="code9708153212215">mean</idp:inline>. </li><li>The shape is the same as that of <idp:inline displayname="code" id="code3284928141915">gamma</idp:inline>.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
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

  - <term>Atlas training series products</term>:
  
    - The data types of `gradOut`, `input`, `mean`, `rstd`, `gamma`, `gradInput`, `gradGammaOut`, and `gradBetaOut` cannot be BFLOAT16.
    - The data types of `mean` and `gradOut` must be the same.

  - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:

    The data types of <idp:inline displayname="code" id="code3426122792515">mean</idp:inline> and <idp:inline displayname="code" id="code442615277252">gradOut</idp:inline> must be the same.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown.

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
      <td rowspan="4">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="4">161001</td>
      <td>The passed gradOut, input, mean, or rstd is a null pointer.</td>
    </tr>
    <tr>
      <td>outputMask[0] is true, and gradInput is a null pointer.</td>
    </tr>
    <tr>
      <td>outputMask[1] is true, and gradGammaOut is a null pointer.</td>
    </tr>
    <tr>
      <td>outputMask[2] is true, and gradBetaOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>The data type of gradOut is not supported.</td>
    </tr>
    <tr>
      <td>The data types of input, mean, gamma, rstd, and gradOut do not meet the constraints specified in the parameter description.</td>
    </tr>
    <tr>
      <td>The length of outputMask is not 3.</td>
    </tr>
    <tr>
      <td>When outputMask[0] is true, the shape of gradInput is different from that of input.</td>
    </tr>
    <tr>
      <td>When outputMask[1] is true, the shape of gradGammaOut is different from that of gamma.</td>
    </tr>
    <tr>
      <td>When outputMask[2] is true, the shape of gradBetaOut is different from that of gamma.</td>
    </tr>
    <tr>
      <td>The value of group is not greater than 0.</td>
    </tr>
    <tr>
      <td>The value of C cannot be exactly divided by that of group.</td>
    </tr>
    <tr>
      <td>The number of elements in input is not equal to N × C × HxW.</td>
    </tr>
    <tr>
      <td>The number of elements in mean is not equal to N × group.</td>
    </tr>
    <tr>
      <td>The number of elements in rstd is not equal to N × group.</td>
    </tr>
    <tr>
      <td>gamma is not a null pointer and the number of elements in gamma is not equal to C.</td>
    </tr>
  </tbody></table>

## aclnnGroupNormBackward

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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnGroupNormBackwardGetWorkspaceSize.</td>
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

- Deterministic computation:
  - **aclnnGroupNormBackward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic computation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

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
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
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

    // 2. Construct inputs and outputs based on the API definition.
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
    // Create a gradOut aclTensor.
    ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an input aclTensor.
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a mean aclTensor.
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a rstd aclTensor.
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a gamma aclTensor.
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    auto outputMask = aclCreateBoolArray(outputMaskData.data(), outputMaskData.size());
    CHECK_RET(outputMask != nullptr, return ACL_ERROR_INTERNAL_ERROR);
    // Create a gradInput aclTensor.
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a gradGammaOut aclTensor.
    ret = CreateAclTensor(
        gradGammaOutHostData, gradGammaOutShape, &gradGammaOutDeviceAddr, aclDataType::ACL_FLOAT, &gradGammaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a gradBetaOut aclTensor.
    ret = CreateAclTensor(
        gradBetaOutHostData, gradBetaOutShape, &gradBetaOutDeviceAddr, aclDataType::ACL_FLOAT, &gradBetaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual host API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnGroupNormBackward.
    ret = aclnnGroupNormBackwardGetWorkspaceSize(
        gradOut, input, mean, rstd, gamma, N, C, HxW, group, outputMask, gradInput, gradGammaOut, gradBetaOut,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnGroupNormBackward.
    ret = aclnnGroupNormBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
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

    // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(gradOut);
    aclDestroyTensor(input);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);
    aclDestroyTensor(gamma);
    aclDestroyTensor(gradInput);
    aclDestroyTensor(gradGammaOut);
    aclDestroyTensor(gradBetaOut);

    // 7. Release device resources. Modify the code based on the API definition.
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
