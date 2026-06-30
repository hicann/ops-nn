# aclnnPreluBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/activation/prelu_grad_update)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     √    |

## Function

Description: Performs backpropagation of [aclnnPrelu](../../prelu/docs/aclnnPrelu_en.md).
The formula of gradInput is as follows:

$$
gradInput_{i,j,...}=
\begin{cases}
gradOutput_{i,j,...}, & if\ self_{i,j,...} > 0 \\
gradOutput_{i,j,...} * weight_{i}, & if\ self_{i,j,...} <= 0
\end{cases}
$$

The formula of gradWeight is as follows:

$$
gradWeight_{j}=\sum_{i,...}
\begin{cases}
0, & if\ self_{i,j,...} > 0 \\
gradOutput_{i,j,...} * self_{i,j,...}, & if\ self_{i,j,...} <= 0
\end{cases}
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnPreluBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnPreluBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnPreluBackwardGetWorkspaceSize(
  const aclTensor*  gradOutput,
  const aclTensor*  self,
  const aclTensor*  weight,
  aclTensor*        gradInput,
  aclTensor*        gradWeight,
  uint64_t*         workspaceSize,
  aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnPreluBackward(
  void             *workspace,
  uint64_t          workspace_size,
   aclOpExecutor   *executor,
   aclrtStream      stream)
```


## aclnnPreluBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1490px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 240px">
  <col style="width: 300px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
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
      <td>gradOutput</td>
      <td>Input</td>
      <td>Gradient value of backpropagation, corresponding to gradOutput in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The dtype must be the same as that of self. </li><li>The shapes of gradOutput and self must meet the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>, and the shape after broadcasting is the same as that of self.</li></ul></td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>0–8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Forward input value of prelu, corresponding to self in the formula.</td>
      <td>Empty tensors are supported.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>0–8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Weight of prelu, corresponding to weight in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The dtype must be the same as that of self. </li><li>If the shape of self has more than one dimension, the shape of weight can be the same as that of self, the values of their second dimensions are the same, and the values of other dimensions of weight are 1. If weight is a one-dimensional tensor, the number of elements is the same as the value of the second dimension of self. </li><li></li>Otherwise, the number of elements of weight is one.</ul></td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>0–8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>Output</td>
      <td>Gradient value of self.</td>
      <td><ul><li>The dtype must be the same as that of self. </li><li>Its shape and the shape of gradOutput must meet the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a>. </li><li>The shape and data type of gradInput are the same as those of self.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0–8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeight</td>
      <td>Output</td>
      <td>Gradient value of weight.</td>
      <td><ul><li>The dtype must be the same as that of self. </li><li>The data type must be the same as that of weight. </li><li>The shape of gradWeight must be the same as that of weight.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0–8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
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
  
   - <term>Atlas training series products</term>: The data type can be FLOAT16 or FLOAT32.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:
  
  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>The passed gradOutput, self, weight, gradInput, or gradWeight is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type of gradOutput, self, weight, gradInput, or gradWeight is not supported.</td>
    </tr>
    <tr>
      <td>The data types of gradOutput, self, weight, gradInput, and gradWeight are different.</td>
    </tr>
    <tr>
      <td>The dimensions of gradOutput, self, weight, gradInput, or gradWeight are greater than eight.</td>
    </tr>
    <tr>
      <td>The number of elements of weight is different from the number of channels of self, or 1.</td>
    </tr>
    <tr>
      <td>When the number of elements of weight is 1, the shape of gradWeight is different from that of weight.</td>
    </tr>
    <tr>
      <td>The shapes of self and gradOutput do not meet the broadcast condition.</td>
    </tr>
  </tbody></table>


## aclnnPreluBackward

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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnPreluBackwardGetWorkspaceSize.</td>
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
  - **aclnnPreluBackward** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_prelu_backward.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
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
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> weightShape = {2};
  std::vector<int64_t> gradOutputShape = {4, 2};
  std::vector<int64_t> gradInputShape = {4, 2};
  std::vector<int64_t> gradWeightShape = {2};

  void* selfDeviceAddr = nullptr;
  void* gradOutputDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  void* gradWeightDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* gradInput = nullptr;
  aclTensor* gradWeight = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> weightHostData = {0.5, 0.5};
  std::vector<float> gradOutputHostData = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<float> gradInputHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> gradWeightHostData = {0, 0};

  // Create a weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT,
  &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gradInput aclTensor.
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gradWeight aclTensor.
  ret = CreateAclTensor(gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnPreluBackward.
  ret = aclnnPreluBackwardGetWorkspaceSize(gradOutput, self, weight, gradInput, gradWeight, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPreluBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnPreluBackward.
  ret = aclnnPreluBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPreluBackward failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto gradInputSize = GetShapeSize(gradInputShape);
  std::vector<float> gradInputResultData(gradInputSize, 0);
  ret = aclrtMemcpy(gradInputResultData.data(), gradInputResultData.size() * sizeof(gradInputResultData[0]), gradInputDeviceAddr, gradInputSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < gradInputSize; i++) {
    LOG_PRINT("gradInput[%ld] is: %f\n", i, gradInputResultData[i]);
  }

  auto gradWeightSize = GetShapeSize(gradWeightShape);
  std::vector<float> gradWeightResultData(gradWeightSize, 0);
  ret = aclrtMemcpy(gradWeightResultData.data(), gradWeightResultData.size() * sizeof(gradWeightResultData[0]), gradWeightDeviceAddr, gradWeightSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < gradWeightSize; i++) {
    LOG_PRINT("gradWeight[%ld] is: %f\n", i, gradWeightResultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(weight);
  aclDestroyTensor(gradInput);
  aclDestroyTensor(gradWeight);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(gradInputDeviceAddr);
  aclrtFree(gradWeightDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
