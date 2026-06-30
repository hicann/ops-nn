# aclnnEluBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/activation/elu_grad_v2)

## Product Support

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     √    |
|  <term>Atlas training series products</term>   |     √    |

## Function

- Description: Performs backward computation of the [aclnnElu](../../elu/docs/aclnnElu&aclnnInplaceElu_en.md) activation function and outputs the gradient of the forward input of the ELU activation function.

- Formula: **x** denotes an element in **selfOrResult**.

  - When **isResult** is **True**:

    $$
    gradInput = gradOutput *
    \begin{cases}
    scale, \quad x > 0\\
    inputScale \ast (x + \alpha \ast scale),  \quad x \leq 0
    \end{cases}
    $$

  - When **isResult** is **False**:

    $$
    gradInput = gradOutput *
    \begin{cases}
    scale, \quad x > 0\\
    inputScale \ast \alpha \ast scale \ast exp(x \ast inputScale), \quad x \leq 0
    \end{cases}
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnEluBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnEluBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnEluBackwardGetWorkspaceSize(
  const aclTensor* gradOutput,
  const aclScalar* alpha,
  const aclScalar* scale,
  const aclScalar* inputScale,
  bool             isResult,
  const aclTensor* selfOrResult,
  aclTensor*       gradInput,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnEluBackward(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnEluBackwardGetWorkspaceSize

- **Parameters**
  
  <table style="undefined;table-layout: fixed; width: 1370px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 200px">
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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
      <tr>
      <td>gradOutput</td>
      <td>Input</td>
      <td>Gradient of the forward output of the ELU activation function, corresponding to gradInput in the formula.</td>
      <td>Empty tensors are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Activation coefficient of the ELU activation function, corresponding to \alpha in the formula.</td>
      <td><ul><li>If isResult is true, \alpha must be greater than or equal to 0. </li><li>The data type must be convertible to FLOAT (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">Conversion Relationship</a>).</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Scaling factor of the ELU activation function, corresponding to scale in the formula.</td>
      <td>The data type must be convertible to FLOAT (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">Conversion Relationship</a>).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>inputScale</td>
      <td>Input</td>
      <td>Input scaling factor of the ELU activation function, corresponding to inputScale in the formula.</td>
      <td>The data type must be convertible to FLOAT (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">Conversion Relationship</a>).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>isResult</td>
      <td>Input</td>
      <td>Whether the input for ELU backward computation is the forward output of the ELU operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>selfOrResult</td>
      <td>Input</td>
      <td><ul><li>Forward output of the ELU activation function when isResult is true. </li><li>Forward input of the ELU activation function when isResult is false.</li></ul></td>
      <td><ul><li>The data type must be the same as that of gradOutput. </li><li>The shape must be the same as that of gradOutput. </li><li>Empty tensors are supported.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
       <tr>
      <td>gradInput</td>
      <td>Output</td>
      <td>Gradient of the forward input of the ELU activation function, that is, the derivative of the input, corresponding to gradOutput in the formula.</td>
      <td><ul><li>The data type must be convertible from that of gradOutput (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">Conversion Relationship</a>). </li><li>The shape must be the same as that of gradOutput.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
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
      <td>Operator executor, containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
  
   - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be FLOAT or FLOAT16.


- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.
  
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
      <td>The gradOutput, alpha, scale, inputScale, selfOrResult, or gradInput parameter is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type of gradOutput or selfOrResult is not supported.</td>
    </tr>
    <tr>
      <td>The data types of gradOutput and selfOrResult are inconsistent.</td>
    </tr>
    <tr>
      <td>The data type of alpha, scale, or inputScale cannot be converted to FLOAT.</td>
    </tr>
    <tr>
      <td>The data type of gradInput is not convertible from that of gradOutput.</td>
    </tr>
    <tr>
      <td>The shapes of gradOutput, selfOrResult, and gradInput do not match.</td>
    </tr>
     <tr>
      <td>gradOutput, selfOrResult, or gradInput has more than eight dimensions.</td>
    </tr>
    <tr>
      <td>When isResult is set to True, alpha is negative.</td>
    </tr>
  </tbody></table>


## aclnnEluBackward

- **Parameters**

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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnEluBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>


- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic computation:
  - **aclnnEluBackward** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_elu_backward.h"

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
  // (Boilerplate) Initialize resources.
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
                    aclDataType dataType, aclTensor** selfOrResult) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  
  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous selfOrResult.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *selfOrResult = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID (deviceId) based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on API definitions.
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> selfOrResultShape = {2, 2};
  std::vector<int64_t> gradInputShape = {2, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfOrResultDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclScalar* alpha = nullptr;
  aclScalar* scale = nullptr;
  aclScalar* inputScale = nullptr;
  aclTensor* selfOrResult = nullptr;
  aclTensor* gradInput = nullptr;
  std::vector<float> gradOutputHostData = {-2, -1, 0, 1};
  std::vector<float> selfOrResultHostData = {-2, -1, 0, 1};
  std::vector<float> gradInputHostData = {0, 0, 0, 0};
  float alphaValue = 1.0f;
  float scaleValue = 1.0f;
  float inputScaleValue = 1.0f;
  bool isResult = true;
  // Create a gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT,
                        &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an alpha aclScalar.
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // Create a scale aclScalar.
  scale = aclCreateScalar(&scaleValue, aclDataType::ACL_FLOAT);
  CHECK_RET(scale != nullptr, return ret);
  // Create an inputScale aclScalar.
  inputScale = aclCreateScalar(&inputScaleValue, aclDataType::ACL_FLOAT);
  CHECK_RET(inputScale != nullptr, return ret);
  // Create a selfOrResult aclTensor.
  ret = CreateAclTensor(selfOrResultHostData, selfOrResultShape, &selfOrResultDeviceAddr, aclDataType::ACL_FLOAT,
                        &selfOrResult);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gradInput aclTensor.
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnEluBackward API call example
  // 3. Call the CANN operator library API. Modify the API as required.
  // Call the first-phase API of aclnnEluBackward.
  ret = aclnnEluBackwardGetWorkspaceSize(gradOutput, alpha, scale, inputScale, isResult, selfOrResult, gradInput,
                                         &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEluBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnEluBackward.
  ret = aclnnEluBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEluBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyScalar(alpha);
  aclDestroyScalar(scale);
  aclDestroyScalar(inputScale);
  aclDestroyTensor(selfOrResult);
  aclDestroyTensor(gradInput);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfOrResultDeviceAddr);
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
