# aclnnBatchNormReduceBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/sync_batch_norm_backward_reduce)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     √    |

## Function

- Description:
  
  Computes the gradient of the BatchNorm operation during backpropagation and performs reduction operations on some intermediate results to optimize the compute efficiency. The calculation result is as follows:
  - Computes the gradient **$\frac{\partial l}{\partial γ}$** of the loss function **l** with respect to the scaling weight **γ**.
  - Computes the gradient **$\frac{\partial l}{\partial β}$** of the loss function **l** with respect to the offset **β**.
  - Deduces the intermediate quantities **sumDy** and **sumDyXmu** required for computing **$\frac{\partial l}{\partial x_i}$**, by using the deviation **d<sub>yi</sub>** of the loss function **l** with respect to the output **y<sub>i</sub>**, where **$\frac{\partial l}{\partial x_i}$** is the gradient of the loss function **l** with respect to the input **x<sub>i</sub>** of the corresponding layer.
  
- Formula:
  
  $$
  gradWeight = \frac{\partial l}{\partial γ} = \sum^m_{i=0} \frac{\partial l}{\partial y_i} \cdot \hat{(x_i)} = \frac{1}{{\sqrt{σ^2_B + eps}}} \cdot \sum^m_{i=0} \frac{\partial l}  {\partial y_i} \cdot (x_i-μ_B)
  $$
  
  $$
  gradBias = \frac{\partial l}{\partial β} = \sum^m_{i=0} \frac{\partial l}{\partial y_i}
  $$
  
  $$
  sumDy = sum(l, y_i) = \displaystyle \sum^m_{i=0} \frac{\partial l}{\partial y_i}
  $$
  
  $$
  sumDyXmu = sum(l, y_i, x_i, μ_B) = \displaystyle \sum^m_{i=0} \frac{\partial l}{\partial y_i} \cdot (x_i-μ_B)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnBatchNormReduceBackwardGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnBatchNormReduceBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnBatchNormReduceBackwardGetWorkspaceSize(
  const aclTensor* gradOut,
  const aclTensor* input,
  const aclTensor* mean,
  const aclTensor* invstd,
  const aclTensor* weight,
  const bool       inputG,
  const bool       weightG,
  const bool       biasG,
  aclTensor*       sumDy,
  aclTensor*       sumDyXmu,
  aclTensor*       gradWeight,
  aclTensor*       gradBias,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnBatchNormReduceBackward(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnBatchNormReduceBackwardGetWorkspaceSize

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
      <td>Gradient tensor, corresponding to `<math ><mfrac><mrow><mi mathvariant="normal">∂</mi><mi>l</mi></mrow>/<mrow><mi mathvariant="normal">∂</mi><mi>y</mi></mrow></mfrac></math>` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type and shape must be the same as those of `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input</td>
      <td>Input</td>
      <td>Input tensor, corresponding to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>By default, the second dimension is the channel axis, and the value of the channel axis cannot be 0.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Mean value, corresponding to `μ<sub>B</sub>` in the formula.</td>
      <td><ul><li>The shape length is the same as the length of the channel axis of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invstd</td>
      <td>Input</td>
      <td>Reciprocal of the standard deviation, corresponding to the reciprocal of the square root of `(σ<sub>B</sub>)<sup>2</sup>+eps` in the formula.</td>
      <td><ul><li>The shape length is the same as the length of the channel axis of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Weight tensor, corresponding to `γ` in the formula.</td>
      <td><ul><li>The shape length is the same as the length of the channel axis of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>inputG</td>
      <td>Input</td>
      <td>Output mask, indicating whether to output `sumDy` and `sumDyXmu`.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightG</td>
      <td>Input</td>
      <td>Output mask, indicating whether to output `gradWeight`.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>biasG</td>
      <td>Input</td>
      <td>Output mask, indicating whether to output `gradBias`.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sumDy</td>
      <td>Output</td>
      <td>Accumulated sum of the forward output gradient `gradOut`, corresponding to `sumDy` in the formula.</td>
      <td><ul><li>Optional. If `inputG` is True, the output is returned. The size of the shape must be the same as the length of the channel axis of `input`. </li><li>The data format must be the same as that of `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sumDyXmu</td>
      <td>Output</td>
      <td>Sum of the product of the forward output gradient `gradOut` and the input centralized data `(x-μ<sub>B</sub>)`, corresponding to `sumDyXmu` in the formula.</td>
      <td><ul><li>Optional. If `inputG` is True, the output is returned. The size of the shape must be the same as the length of the channel axis of `input`. </li><li>The data format must be the same as that of `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeight</td>
      <td>Output</td>
      <td>Gradient of the scaling parameter, corresponding to `gradWeight` in the formula.</td>
      <td><ul><li>Optional. If `weightG` is True, the output is returned. The size of the shape must be the same as the length of the channel axis of `input`. </li><li>The data format must be the same as that of `gradOut`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBias</td>
      <td>Output</td>
      <td>Gradient of the bias parameter, corresponding to `gradBias` in the formula.</td>
      <td><ul><li>Optional. If `biasG` is True, the output is returned. The size of the shape must be the same as the length of the channel axis of `input`. </li><li>The data format must be the same as that of `gradOut`.</li></ul></td>
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

  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: The data types of `gradOut`, `input`, `mean`, `invstd`, `weight`, `sumDy`, `sumDyXmu`, `gradWeight`, and `gradBias` cannot be BFLOAT16.

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
      <td>The data type of input, meanAll, invstdAll, mean, invstd, runningMean, runningVar, or counts is not supported.</td>
    </tr>
    <tr>
      <td>When inputG is true, sumDy or sumDyXmu is a null pointer.</td>
    </tr>
    <tr>
      <td>When weightG is true, gradWeight is a null pointer.</td>
    </tr>
    <tr>
      <td>When biasG is true, gradBias is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>The data type of gradOut, input, mean, invstd, weight, sumDy, sumDyXmu, gradWeight, or gradBias is not supported.</td>
    </tr>
    <tr>
      <td>The data type of gradOut must be the same as that of input.</td>
    </tr>
    <tr>
      <td>When inputG is true, the data type or format of sumDy or sumDyXmu is not supported.</td>
    </tr>
    <tr>
      <td>When weightG is true, the data type or format of gradWeight is not supported.</td>
    </tr>
    <tr>
      <td>When biasG is true, the data type or format of gradBias is not supported.</td>
    </tr>
    <tr>
      <td>The data formats of gradOut and input are inconsistent.</td>
    </tr>
    <tr>
      <td>The shape of gradOut or input is greater than 8D.</td>
    </tr>
    <tr>
      <td>The shape of gradOut or input is less than 2D.</td>
    </tr>
    <tr>
      <td>The size of the channel axis of input is 0.</td>
    </tr>
    <tr>
      <td>When inputG is true, the size of sumDy or sumDyXmu is inconsistent with that of the channel axis of input.</td>
    </tr>
    <tr>
      <td>When weightG is true, the size of gradWeight is inconsistent with that of the channel axis of input.</td>
    </tr>
    <tr>
      <td>When biasG is true, the size of gradBias is inconsistent with that of the channel axis of input.</td>
    </tr>
  </tbody></table>

## aclnnBatchNormReduceBackward

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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API **aclnnBatchNormReduceBackwardGetWorkspaceSize**.</td>
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

- If any input is an empty tensor, the output is an empty tensor.
- Deterministic compute:
  - **aclnnBatchNormReduceBackward** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm_backward_reduce.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
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
  std::vector<int64_t> gradOutShape = {4, 2};
  std::vector<int64_t> inputShape = {4, 2};
  std::vector<int64_t> meanShape = {2};
  std::vector<int64_t> invstdShape = {2};
  std::vector<int64_t> weightShape = {2};
  std::vector<int64_t> sumDyShape = {2};
  std::vector<int64_t> sumDyXmuShape = {2};
  std::vector<int64_t> gradWeightShape = {2};
  std::vector<int64_t> gradBiasShape = {2};

  void* gradOutDeviceAddr = nullptr;
  void* inputDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* invstdDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* sumDyDeviceAddr = nullptr;
  void* sumDyXmuDeviceAddr = nullptr;
  void* gradWeightDeviceAddr = nullptr;
  void* gradBiasDeviceAddr = nullptr;

  aclTensor* gradOut = nullptr;
  aclTensor* input = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* invstd = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* sumDy = nullptr;
  aclTensor* sumDyXmu = nullptr;
  aclTensor* gradWeight = nullptr;
  aclTensor* gradBias = nullptr;

  std::vector<float> gradOutHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> inputHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> meanHostData = {1, 1};
  std::vector<float> invstdHostData = {1, 1};
  std::vector<float> weightHostData = {1, 1};
  std::vector<float> sumDyHostData = {1, 1};
  std::vector<float> sumDyXmuHostData = {1, 1};
  std::vector<float> gradWeightHostData = {1, 1};
  std::vector<float> gradBiasHostData = {1, 1};

  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(invstdHostData, invstdShape, &invstdDeviceAddr, aclDataType::ACL_FLOAT, &invstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  bool inputG = true;
  bool weightG = true;
  bool biasG = true;

  ret = CreateAclTensor(sumDyHostData, sumDyShape, &sumDyDeviceAddr, aclDataType::ACL_FLOAT, &sumDy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sumDyXmuHostData, sumDyXmuShape, &sumDyXmuDeviceAddr, aclDataType::ACL_FLOAT, &sumDyXmu);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradBiasHostData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBatchNormReduceBackward API call example
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  // Call the first-phase API of aclnnBatchNormReduceBackward.
  ret = aclnnBatchNormReduceBackwardGetWorkspaceSize(gradOut, input, mean, invstd, weight,
                                                     inputG, weightG, biasG,
                                                     sumDy, sumDyXmu, gradWeight, gradBias,
                                                     &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormReduceBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnBatchNormReduceBackward.
  ret = aclnnBatchNormReduceBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormReduceBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  PrintOutResult(sumDyShape, &sumDyDeviceAddr);
  PrintOutResult(sumDyXmuShape, &sumDyXmuDeviceAddr);
  PrintOutResult(gradWeightShape, &gradWeightDeviceAddr);
  PrintOutResult(gradBiasShape, &gradBiasDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(gradOut);
  aclDestroyTensor(input);
  aclDestroyTensor(mean);
  aclDestroyTensor(invstd);
  aclDestroyTensor(weight);
  aclDestroyTensor(sumDy);
  aclDestroyTensor(sumDyXmu);
  aclDestroyTensor(gradWeight);
  aclDestroyTensor(gradBias);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(inputDeviceAddr);
  aclrtFree(meanDeviceAddr);
  aclrtFree(invstdDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(sumDyDeviceAddr);
  aclrtFree(sumDyXmuDeviceAddr);
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
