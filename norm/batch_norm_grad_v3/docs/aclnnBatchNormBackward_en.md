# aclnnBatchNormBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/batch_norm_grad_v3)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     √    |

## Function

- Description: Performs backpropagation of [aclnnBatchNorm](../../batch_norm_v3/docs/aclnnBatchNorm_en.md). It is used to compute the gradient of the input tensor, so that the model parameters can be updated during backpropagation.

- Formula:

  - When **training** is **true**:
  
    $$
    gradInput = \frac{weight}{ n{\sqrt{saveVar + eps}} }(n * gradOut - \sum^m_{i=0}{gradOut} - \frac{x-saveMean}{ {\sqrt{saveVar + eps}} }\sum^m_{i=0}({gradOut} *\frac{x-saveMean}{ {\sqrt{saveVar + eps}} } ))
    $$

    $$
    gradWeight = \sum^m_{i=0}[{gradOut} * (x - saveMean)] * \frac{1}{ {\sqrt{saveVar + eps}} } 
    $$

    $$
    gradBias = \sum^m_{i=0}{gradOut} 
    $$


  - When **training** is **false**:

    $$
    gradInput = gradOut * \frac{1}{ {\sqrt{runningVar + eps}} } * weight
    $$

    $$
    gradWeight = \sum^m_{i=0}[{gradOut} * (x - runningMean)] * \frac{1}{ {\sqrt{runningVar + eps}} } 
    $$

    $$
    gradBias = \sum^m_{i=0}{gradOut} 
    $$

  
## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnBatchNormBackwardGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnBatchNormBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnBatchNormBackwardGetWorkspaceSize(
  const aclTensor    *gradOut,
  const aclTensor    *input,
  const aclTensor    *weight,
  const aclTensor    *runningMean,
  const aclTensor    *runningVar,
  const aclTensor    *saveMean,
  const aclTensor    *saveInvstd,
  bool                training,
  double              eps,
  const aclBoolArray *outputMask,
  aclTensor          *gradInput,
  aclTensor          *gradWeight,
  aclTensor          *gradBias,
  uint64_t           *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnBatchNormBackward(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnBatchNormBackwardGetWorkspaceSize

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
      <td>Gradient tensor, corresponding to `gradOut` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The supported shapes and formats are as follows: 2D (NC), 3D (NCL), 4D (NCHW), 5D (NCDHW), and 6D to 8D (ND, where the second dimension is fixed to the channel axis).</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NC, NCL, NCHW, NHWC, NCDHW, NDHWC, ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input</td>
      <td>Input</td>
      <td>Forward input tensor, corresponding to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type, shape, and data format must be the same as those of `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NC, NCL, NCHW, NHWC, NCDHW, NDHWC, ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Weight tensor, corresponding to `weight` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape length is the same as the length of the channel axis of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningMean</td>
      <td>Input</td>
      <td>Mean value computed during training, corresponding to `runningMean` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape length is the same as the length of the channel axis of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningVar</td>
      <td>Input</td>
      <td>Variance computed during training, corresponding to `runningVar` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The value is a non-negative number. </li><li>The shape length is the same as the length of the channel axis of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>saveMean</td>
      <td>Input</td>
      <td>Saved mean value, corresponding to `saveMean` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape length is the same as the length of the channel axis of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>saveInvstd</td>
      <td>Input</td>
      <td>Reciprocal of the standard deviation, corresponding to the reciprocal of the square root of (Var(x) + eps) in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The value is a non-negative number. </li><li>The shape length is the same as the length of the channel axis of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>training</td>
      <td>Input</td>
      <td>Whether the scenario is a training scenario, corresponding to `training` in the formula.</td>
      <td>true indicates a training scenario, and false indicates an inference scenario.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Input</td>
      <td>Value to be added to the variance to avoid division by zero. It corresponds to `eps` in the formula.</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMask</td>
      <td>Input</td>
      <td>Output mask.</td>
      <td>The size is 3. This parameter indicates whether to output `gradInput`, `gradWeight`, and `gradBias`. If **true**, they are output; otherwise, null is returned in the corresponding positions.</td>
      <td>BOOLARRAY</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>Output</td>
      <td>Gradient of the input tensor, corresponding to `gradInput` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>Optional. If outputMask[0] is true, this output is required. Otherwise, this output is not required. </li><li>The data type, shape, and data format must be the same as those of <idp:inline displayname="code" id="code11608153105416">gradOut</idp:inline>.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NC, NCL, NCHW, NHWC, NCDHW, NDHWC, ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeight</td>
      <td>Output</td>
      <td>Gradient of the scaling parameter, corresponding to `gradWeight` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>Optional. If outputMask[1] is true, this output is required. Otherwise, this output is not required. </li><li>The length is the same as the length of the channel axis in the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBias</td>
      <td>Output</td>
      <td>Gradient of the bias parameter, corresponding to `gradBias` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>Optional. If outputMask[2] is True, the output is required. Otherwise, the output is not required. </li><li>The shape length is the same as the length of the channel axis of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
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

  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>:
    - The data types of the `gradOut`, `input`, `weight`, `runningMean`, `runningVar`, `saveMean`, `saveInvstd`, `gradInput`, `gradWeight`, and `gradBias` parameters cannot be BFLOAT16.
    - The data types of the `weight`, `runningMean`, `runningVar`, `saveMean`, `saveInvstd`, `gradWeight`, and `gradBias` parameters are the same as that of `gradOut`.
    - The data formats of the `gradOut`, `input`, and `gradInput` parameters cannot be NHWC and NDHWC.

  - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:
    - The data types of the `weight`, `runningMean`, `runningVar`, `saveMean`, `saveInvstd`, `gradWeight`, and `gradBias` parameters are the same as that of `gradOut`.
    - The data formats of the `gradOut`, `input`, and `gradInput` parameters cannot be NHWC or NDHWC.

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
      <td>The passed gradOut or input is a null pointer.</td>
    </tr>
    <tr>
      <td>outputMask[0] is true, and gradInput is a null pointer.</td>
    </tr>
    <tr>
      <td>outputMask[1] is true, and gradWeight is a null pointer.</td>
    </tr>
    <tr>
      <td>outputMask[2] is true, and gradBias is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>The input, gradOut, data type, data format, and shape are not supported.</td>
    </tr>
    <tr>
      <td>weight, runningMean, runningVar, saveMean, and saveInvstd are not empty. The data type, data format, and shape are not supported.</td>
    </tr>
    <tr>
      <td>The length of outputMask is not 3.</td>
    </tr>
    <tr>
      <td>outputMask[0] is true, and the data type, data format, and shape of gradInput are not supported.</td>
    </tr>
    <tr>
      <td>outputMask[1] is true, and the data type, data format, and shape of gradWeight are not supported.</td>
    </tr>
    <tr>
      <td>outputMask[2] is true, and the data type, data format, and shape of gradBias are not supported.</td>
    </tr>
    <tr>
      <td>The shape lengths of weight, runningMean, runningVar, saveMean, saveInvstd, gradWeight (not empty), and gradBias (not empty) are not equal to the length of the channel axis in the input shape.</td>
    </tr>
    <tr>
      <td>The data formats of input, gradOut, and gradInput (not empty) are inconsistent.</td>
    </tr>
    <tr>
      <td>The data types of input, gradOut, and gradInput (not empty) are inconsistent.</td>
    </tr>
    <tr>
      <td>The shapes of input, gradOut, and gradInput (not empty) are inconsistent, or the shape has more than 8 dimensions or fewer than 2 dimensions.</td>
    </tr>
  </tbody></table>

## aclnnBatchNormBackward

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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API aclnnBatchNormBackwardGetWorkspaceSize.</td>
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

- Deterministic compute:
  - **aclnnBatchNormBackward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm_backward.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> gradOutShape = {1, 2, 4};
  std::vector<int64_t> selfShape = {1, 2, 4};
  std::vector<int64_t> weightShape = {2};
  std::vector<int64_t> rMeanShape = {2};
  std::vector<int64_t> rVarShape = {2};
  std::vector<int64_t> sMeanShape = {2};
  std::vector<int64_t> sVarShape = {2};
  std::vector<int64_t> gradInShape = {1, 2, 4};
  std::vector<int64_t> gradWeightShape = {2};
  std::vector<int64_t> gradBiasShape = {2};
  void* gradOutDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* rMeanDeviceAddr = nullptr;
  void* rVarDeviceAddr = nullptr;
  void* sMeanDeviceAddr = nullptr;
  void* sVarDeviceAddr = nullptr;
  void* outMaskDeviceAddr = nullptr;
  void* gradInDeviceAddr = nullptr;
  void* gradWeightDeviceAddr = nullptr;
  void* gradBiasDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* self = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* rMean = nullptr;
  aclTensor* rVar = nullptr;
  aclTensor* sMean = nullptr;
  aclTensor* sVar = nullptr;
  aclBoolArray* outMask = nullptr;
  aclTensor* gradIn = nullptr;
  aclTensor* gradWeight = nullptr;
  aclTensor* gradBias = nullptr;
  std::vector<float> gradOutHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> weightHostData = {1, 1};
  std::vector<float> rMeanHostData = {0, 0};
  std::vector<float> rVarHostData = {1, 1};
  std::vector<float> sMeanHostData = {0, 0};
  std::vector<float> sVarHostData = {1, 1};
  std::vector<float> gradInHostData(8, 0);
  std::vector<float> gradWeightHostData(2, 0);
  std::vector<float> gradBiasHostData(2, 0);;
  bool training = true;
  double eps = 1e-5;
  // Create a gradOut aclTensor.
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an rMean aclTensor.
  ret = CreateAclTensor(rMeanHostData, rMeanShape, &rMeanDeviceAddr, aclDataType::ACL_FLOAT, &rMean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an rVar aclTensor.
  ret = CreateAclTensor(rVarHostData, rVarShape, &rVarDeviceAddr, aclDataType::ACL_FLOAT, &rVar);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an sMean aclTensor.
  ret = CreateAclTensor(sMeanHostData, sMeanShape, &sMeanDeviceAddr, aclDataType::ACL_FLOAT, &sMean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an sVar aclTensor.
  ret = CreateAclTensor(sVarHostData, sVarShape, &sVarDeviceAddr, aclDataType::ACL_FLOAT, &sVar);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an outMask aclBoolArray.
  bool maskData[3] = {true, true, true};
  outMask = aclCreateBoolArray(&(maskData[0]), 3);
  // Create a gradIn aclTensor.
  ret = CreateAclTensor(gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT, &gradIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gradWeight aclTensor.
  ret = CreateAclTensor(gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gradBias aclTensor.
  ret = CreateAclTensor(gradBiasHostData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBatchNormBackward call example
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  // Call the first-phase API of aclnnBatchNormBackward.
  ret = aclnnBatchNormBackwardGetWorkspaceSize(gradOut, self, weight, rMean, rVar, sMean, sVar, training, eps, outMask, gradIn, gradWeight, gradBias, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnBatchNormBackward.
  ret = aclnnBatchNormBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(gradInShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(gradOut);
  aclDestroyTensor(self);
  aclDestroyTensor(weight);
  aclDestroyTensor(rMean);
  aclDestroyTensor(rVar);
  aclDestroyTensor(sMean);
  aclDestroyTensor(sVar);
  aclDestroyBoolArray(outMask);
  aclDestroyTensor(gradIn);
  aclDestroyTensor(gradWeight);
  aclDestroyTensor(gradBias);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(rMeanDeviceAddr);
  aclrtFree(rVarDeviceAddr);
  aclrtFree(sMeanDeviceAddr);
  aclrtFree(sVarDeviceAddr);
  aclrtFree(outMaskDeviceAddr);
  aclrtFree(gradInDeviceAddr);
  aclrtFree(gradWeightDeviceAddr);
  aclrtFree(gradBiasDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
