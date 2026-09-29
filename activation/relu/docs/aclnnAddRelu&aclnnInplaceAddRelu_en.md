# aclnnAddRelu&aclnnInplaceAddRelu

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/activation/relu)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     √    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     √    |

## Function

- Description: Performs the addition operation and activates the result.
- Formula:

  $$
  out_i = self_i+alpha \times other_i
  $$

  $$
  relu(self) = \begin{cases} self, & self\gt 0 \\ 0, & self\le 0 \end{cases}
  $$

## Prototype

- aclnnAddRelu and aclnnInplaceAddRelu implement the same function. The differences are as follows. Select a proper operator based on the actual scenario.

  - aclnnAddRelu: You need to create an output tensor object to store the computation result.
  - aclnnInplaceAddRelu: You do not need to create an output tensor object. The computation result is stored in the memory of the input tensor.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnAddReluGetWorkspaceSize** or **aclnnInplaceAddReluGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnAddRelu** or **aclnnInplaceAddRelu** is called to perform computation.

  ```Cpp
  aclnnStatus aclnnAddReluGetWorkspaceSize(
    const aclTensor   *self,
    const aclTensor   *other,
    aclScalar         *alpha,
    aclTensor         *out,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
  ```

  ```Cpp
  aclnnStatus aclnnAddRelu(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceAddReluGetWorkspaceSize(
    aclTensor         *selfRef,
    const aclTensor   *other,
    aclScalar         *alpha,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceAddRelu(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
  ```

## aclnnAddReluGetWorkspaceSize

- **Parameters:**
  
  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
    <col style="width: 249px">
    <col style="width: 121px">
    <col style="width: 164px">
    <col style="width: 253px">
    <col style="width: 262px">
    <col style="width: 148px">
    <col style="width: 135px">
    <col style="width: 146px">
    </colgroup>
    <thead>
      <tr>
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
      <td>self (aclTensor*) </td>
      <td>Input</td>
      <td>The input self in the formula indicates the target tensor to be converted.</td>
      <td><ul><li>The shape must be broadcastable with that of `other` (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>). </li><li>The data types of `self` and `other` must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT16, INT32, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
      </tr>
      <tr>
      <td>other (aclTensor*) </td>
      <td>Input</td>
      <td>Input `other` in the formula.</td>
      <td><ul><li>The shape must be broadcastable with that of `self` (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>). </li><li>The data types of `self` and `other` must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT16, INT32, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
      </tr>
      <tr>
      <td>alpha (aclScalar*) </td>
      <td>Input</td>
      <td>alpha in the formula.</td>
      <td>Its data type can be cast to the type promoted from `self` and `other`.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT16, INT32, INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>out (aclTensor*) </td>
      <td>Output</td>
      <td>`out` in the formula.</td>
      <td>The data type must be convertible from the deduced data type of `self` and `other`. The shape must be the broadcast result of `self` and `other`.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT16, INT32, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
      </tr>
      <tr>
      <td>workspaceSize (uint64_t*) </td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>executor (aclOpExecutor**) </td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
    </tbody></table>

    - For the <term>Atlas training products</term>, the data types of the `self`, `other`, `alpha` and `out` parameters do not support BFLOAT16.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
    <col style="width: 267px">
    <col style="width: 124px">
    <col style="width: 775px">
    </colgroup>
    <thead>
      <tr>
        <th>Return code</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>`self`, `other`, `alpha`, or `out` is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="10">161002</td>
        <td>The data type of `self` or `other` is not supported.</td>
      </tr>
      <tr>
        <td>Type promotion between `self` and `other` cannot be performed.</td>
      </tr>
      <tr>
        <td>The deduced data type cannot be converted to that of `out`.</td>
      </tr>
      <tr>
        <td>The shapes of `self` and `other` are not broadcastable.</td>
      </tr>
      <tr>
        <td>The data shape of `alpha` cannot be cast to the data type promoted from `self` and `other`.</td>
      </tr>
    </tbody>
    </table>

## aclnnAddRelu

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnAddReluGetWorkspaceSize API.</td>
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

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceAddReluGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
    <col style="width: 249px">
    <col style="width: 121px">
    <col style="width: 164px">
    <col style="width: 253px">
    <col style="width: 262px">
    <col style="width: 148px">
    <col style="width: 135px">
    <col style="width: 146px">
    </colgroup>
    <thead>
      <tr>
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
      <td>selfRef (aclTensor*) </td>
      <td>Input | Output</td>
      <td>In the formula, self and out indicate the target tensor to be converted.</td>
      <td>The data types of `self` and `other` must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).and must be convertible after deduction.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT16, INT32, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
      </tr>
      <tr>
      <td>other (aclTensor*) </td>
      <td>Input</td>
      <td>Input `other` in the formula.</td>
      <td><ul><li>The shape must be in the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with selfRef. </li><li>The data types of `self` and selfRef must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT16, INT32, INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
      </tr>
      <tr>
      <td>alpha (aclScalar*) </td>
      <td>Input</td>
      <td>alpha in the formula.</td>
      <td>The data type can be converted to the data type deduced from selfRef and other.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT16, INT32, INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>workspaceSize (uint64_t*) </td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>executor (aclOpExecutor**) </td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
    </tbody></table>

    - For the <term>Atlas training products</term>, the data types of the `selfRef`, `other` and `alpha` parameters do not support BFLOAT16.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
    <col style="width: 267px">
    <col style="width: 124px">
    <col style="width: 775px">
    </colgroup>
    <thead>
      <tr>
        <th>Return Code</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>The input selfRef, other, or alpha is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="10">161002</td>
        <td>The data type of `selfRef` or `other` is not supported.</td>
      </tr>
      <tr>
        <td>Data type deduction between selfRef and other cannot be performed.</td>
      </tr>
      <tr>
        <td>The deduced data type cannot be converted to the type of selfRef.</td>
      </tr>
      <tr>
        <td>The shapes of `selfRef` and `other` are not broadcastable.</td>
      </tr>
      <tr>
        <td>The data shape of alpha cannot be converted to the data type deduced from selfRef and other.</td>
      </tr>
    </tbody>
    </table>

## aclnnInplaceAddRelu

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnInplaceAddReluGetWorkspaceSize.</td>
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

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - **aclnnAddRelu&aclnnInplaceAddRelu** defaults to a deterministic implementation.

- For the scenario where the data type of **selfRef** is INT8 and that of **other** is INT32:
    The cast operator has a precision issue when converting the INT32 type to the INT8 type (see [aclnnCast](https://gitcode.com/cann/ops-math/blob/master/math/cast/docs/aclnnCast.md)). In this scenario, the output result precision cannot be ensured.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_relu.h"

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
  // Call aclrtMemcpy to copy the data from the host to the device.
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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> otherShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> otherHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(8, 0);
  float alphaValue = 1.2f;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an alpha aclScalar.
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  
  // aclnnAddRelu API call example 
  // 3. Call the CANN operator library API.
  // Call the first-phase API of aclnnAddRelu.
  ret = aclnnAddReluGetWorkspaceSize(self, other, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddReluGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAddRelu.
  ret = aclnnAddRelu(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRelu failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

    
  // aclnnInplaceAddRelu API call example 
  // 3. Call the CANN operator library API.
  LOG_PRINT("\ntest aclnnInplaceAddRelu\n");
  // Call the first-phase API of aclnnInplaceAddRelu.
  ret = aclnnInplaceAddReluGetWorkspaceSize(self, other, alpha, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddReluGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize calculated by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceAddRelu.
  ret = aclnnInplaceAddRelu(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddRelu failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }  
     
    
  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
