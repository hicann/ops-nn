# aclnnBatchNormElemtBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/sync_batch_norm_backward_elemt)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     √    |

## Function

- Description: Performs backpropagation of [aclnnBatchNormElemt](../../batch_norm_elemt/docs/aclnnBatchNormElemt_en.md). It is used to compute the element-level gradient of the input tensor, so that the model parameters can be updated during backpropagation.
- Formula:

  $$
  gradInput = ({gradOut} - \frac{sumDy}{ {counter}}) - ((input - mean) * (invstd^{2} *   (\frac{sumDyXmu}{ {counter}}))) * invstd * weight
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnBatchNormElemtBackwardGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnBatchNormElemtBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnBatchNormElemtBackwardGetWorkspaceSize(
  const aclTensor* gradOut,
  const aclTensor* input,
  const aclTensor* mean,
  const aclTensor* invstd,
  const aclTensor* weight,
  const aclTensor* sumDy,
  const aclTensor* sumDyXmu,
  aclTensor*       counter,
  aclTensor*       gradInput,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnBatchNormElemtBackward(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnBatchNormElemtBackwardGetWorkspaceSize

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
      <td>Differentiation of the forward output, corresponding to `gradOut` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as that of `input`. </li><li>The second dimension is fixed to the channel axis.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input</td>
      <td>Input</td>
      <td>Input for BatchNorm computation, corresponding to `input` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The second dimension is fixed to the channel axis, and the size of the channel axis cannot be 0.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Mean value of the input data, corresponding to `mean` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape length is the same as the length of the channel axis of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invstd</td>
      <td>Input</td>
      <td>Reciprocal of the standard deviation of the input data, corresponding to `invstd` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape length is the same as the length of the channel axis of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Weight tensor, corresponding to `weight` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>Optional. </li><li>The shape length is the same as the length of the channel axis of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sumDy</td>
      <td>Input</td>
      <td>Average value of the sample mean sum of the output gradient, corresponding to `sumDy` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape length is the same as the length of the channel axis of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sumDyXmu</td>
      <td>Input</td>
      <td>Average value of the product of the sample mean sum and the input gradient, corresponding to `sumDyXmu` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape length is the same as the length of the channel axis of the input parameter `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>counter</td>
      <td>Input</td>
      <td>Number of input data records, corresponding to `counter` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li></ul></td>
      <td>INT32, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>Output</td>
      <td>Gradient of the input tensor, corresponding to `gradInput` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as that of `input`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
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

  - <term>Atlas training series products</term>: The data types of `gradOut`, `input`, `mean`, `invstd`, `weight`, `sumDy`, `sumDyXmu`, and `gradInput` cannot be BFLOAT16.

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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed gradOut, input, mean, invstd, sumDy, sumDyXmu, counter, or gradInput is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The data type of gradOut, input, mean, invstd, sumDy, sumDyXmu, counter, or gradInput is not supported.</td>
    </tr>
    <tr>
      <td>When weight is not a null pointer, the data type of weight is not supported.</td>
    </tr>
    <tr>
      <td>The data format of gradOut, input, or gradInput is not supported.</td>
    </tr>
    <tr>
      <td>The input has fewer than 2 dimensions.</td>
    </tr>
    <tr>
      <td>The shape of input, gradOut, gradInput, or counter is greater than 8D.</td>
    </tr>
    <tr>
      <td>The size of the channel axis of input is 0.</td>
    </tr>
    <tr>
      <td>The shape of gradOut or gradInput is inconsistent with that of input.</td>
    </tr>
    <tr>
      <td>The shape of mean, invstd, sumDy, or sumDyXmu is inconsistent with the channel axis of input.</td>
    </tr>
    <tr>
      <td>When weight is not a null pointer, the shape of weight is inconsistent with the channel axis of input.</td>
    </tr>
  </tbody></table>

## aclnnBatchNormElemtBackward

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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API aclnnBatchNormElemtBackwardGetWorkspaceSize.</td>
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
  - **aclnnBatchNormElemtBackward** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm_elemt_backward.h"

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
  std::vector<int64_t> inputShape = {1, 2, 4};
  std::vector<int64_t> meanShape = {2};
  std::vector<int64_t> invstdShape = {2};
  std::vector<int64_t> weightShape = {2};
  std::vector<int64_t> sumDyShape = {2};
  std::vector<int64_t> sumDyXmuShape = {2};
  std::vector<int64_t> counterShape = {2};
  std::vector<int64_t> gradInputShape = {1, 2, 4};
  void* gradOutDeviceAddr = nullptr;
  void* inputDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* invstdDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* sumDyDeviceAddr = nullptr;
  void* sumDyXmuDeviceAddr = nullptr;
  void* counterDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* input = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* invstd = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* sumDy = nullptr;
  aclTensor* sumDyXmu = nullptr;
  aclTensor* counter = nullptr;
  aclTensor* gradInput = nullptr;
  std::vector<float> gradOutHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> inputHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> meanHostData = {0, 0};
  std::vector<float> invstdHostData = {1, 1};
  std::vector<float> weightHostData = {1, 1};
  std::vector<float> sumDyHostData = {0, 0};
  std::vector<float> sumDyXmuHostData = {1, 1};
  std::vector<float> counterHostData = {5, 5};
  std::vector<float> gradInputHostData(8, 0);

  // Create a gradOut aclTensor.
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an input aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a mean aclTensor.
  ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an invstd aclTensor.
  ret = CreateAclTensor(invstdHostData, invstdShape, &invstdDeviceAddr, aclDataType::ACL_FLOAT, &invstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a sumDy aclTensor.
  ret = CreateAclTensor(sumDyHostData, sumDyShape, &sumDyDeviceAddr, aclDataType::ACL_FLOAT, &sumDy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a sumDyXmu aclTensor.
  ret = CreateAclTensor(sumDyXmuHostData, sumDyXmuShape, &sumDyXmuDeviceAddr, aclDataType::ACL_FLOAT, &sumDyXmu);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a counter aclTensor.
  ret = CreateAclTensor(counterHostData, counterShape, &counterDeviceAddr, aclDataType::ACL_FLOAT, &counter);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gradInput aclTensor.
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBatchNormElemtBackward API call example
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  // Call the first-phase API of aclnnBatchNormElemtBackward.
  ret = aclnnBatchNormElemtBackwardGetWorkspaceSize(gradOut, input, mean, invstd, weight, sumDy, sumDyXmu, counter, gradInput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormElemtBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnBatchNormElemtBackward.
  ret = aclnnBatchNormElemtBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormElemtBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(gradOut);
  aclDestroyTensor(input);
  aclDestroyTensor(weight);
  aclDestroyTensor(mean);
  aclDestroyTensor(invstd);
  aclDestroyTensor(sumDy);
  aclDestroyTensor(sumDyXmu);
  aclDestroyTensor(counter);
  aclDestroyTensor(gradInput);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(inputDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(meanDeviceAddr);
  aclrtFree(invstdDeviceAddr);
  aclrtFree(sumDyDeviceAddr);
  aclrtFree(sumDyXmuDeviceAddr);
  aclrtFree(counterDeviceAddr);
  aclrtFree(gradInputDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
