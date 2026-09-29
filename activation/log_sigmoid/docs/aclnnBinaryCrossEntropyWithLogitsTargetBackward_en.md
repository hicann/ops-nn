# aclnnBinaryCrossEntropyWithLogitsTargetBackward

[📄 View Source Code](https://gitcode.com/cann/ops-nn/tree/master/activation/log_sigmoid)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

Computes logits of the input self, and performs backpropagation of the [BECLoss](../../../loss/sigmoid_cross_entropy_with_logits_v2/docs/aclnnBinaryCrossEntropyWithLogits_en.md) with respect to the target label value **target**.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnBinaryCrossEntropyWithLogitsTargetBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize(
  const aclTensor *gradOutput, 
  const aclTensor *self, 
  const aclTensor *target, 
  const aclTensor *weightOptional, 
  const aclTensor *posWeightOptional, 
  int64_t          reduction, 
  aclTensor       *gradTarget, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnBinaryCrossEntropyWithLogitsTargetBackward(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize

- **Parameters:**

  <table class="tg" style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 350px">
  <col style="width: 220px">
  <col style="width: 115px">
  <col style="width: 120px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter Name</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Usage Description</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">gradOutput (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Gradient value of the previous step before the network backpropagation.</td>
      <td class="tg-0pky">The shape must be <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcastable</a> with that of `self`</td>.
      <td class="tg-0pky">FLOAT16, FLOAT, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">self (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Calculation result of the previous layer in the forward propagation.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">FLOAT16, FLOAT, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">target (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Label value.</td>
      <td class="tg-0pky">The shape must be the same as that of self.</td>
      <td class="tg-0pky">FLOAT16, FLOAT, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">Same as self.</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">weightOptional (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Weight of binary cross entropy.</td>
      <td class="tg-0pky">The shape must be <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcastable</a> to that of `self`. If weightOptional is empty, a tensor of all 1s is created based on the shape of `self`.</td>
      <td class="tg-0pky">FLOAT16, FLOAT, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">posWeightOptional (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Weight of the positive class.</td>
      <td class="tg-0pky">The shape can be broadcast to the shape of self. If posWeightOptional is empty, a tensor filled with 1s is created based on the shape of self.</td>
      <td class="tg-0pky">FLOAT16, FLOAT, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">reduction (int64_t) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Reduction operation performed on the result of the backward gradient calculation of binary cross entropy.</td>
      <td class="tg-0pky">Only 0, 1, and 2 are supported. 0 indicates that no operation is performed. 1 indicates that the average value of the result is used. 2 indicates that the sum of the result is used.</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">gradTarget (aclTensor*) </td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Stores the gradient calculation result.</td>
      <td class="tg-0lax">dtype of the input tensor after derivation, which can be converted to the dtype of gradTarget.</td>
      <td class="tg-0lax">FLOAT16, FLOAT, BFLOAT16</td>
      <td class="tg-0lax">Same as self.</td>
      <td class="tg-0lax">Same as self.</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the size of the workspace to be allocated on the device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor (aclOpExecutor**) </td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the operator executor, including the operator calculation process.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table class="tg" style="undefined;table-layout: fixed; width: 950px"><colgroup>
  <col style="width: 276px">
  <col style="width: 132px">
  <col style="width: 542px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Returns</th>
      <th class="tg-0pky">Error Code</th>
      <th class="tg-0pky">Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">The input gradOutput, self, target, and gradTarget are null pointers. </td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="7">161002</td>
      <td class="tg-0pky">The data types and formats of gradOutput, self, target, and gradTarget are not supported.</td>
    </tr>
    <tr>
      <td class="tg-0pky">When weightOptional and posWeightOptional are not null pointers, their data types and formats are not supported.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shapes of self, target, and gradTarget are inconsistent.</td>
    </tr>
    <tr>
      <td class="tg-0pky">When weightOptional and posWeightOptional are not null pointers, their shapes cannot be broadcast to the shape of self.</td>
    </tr>
    <tr>
      <td class="tg-0pky">gradOutput cannot be broadcast to the shape of self.</td>
    </tr>
    <tr>
      <td class="tg-0lax">The value of reduction is not 0, 1, or 2.</td>
    </tr>
    <tr>
      <td class="tg-0lax">The dimensions of gradOutput, self, target, and gradTarget are greater than 8.</td>
    </tr>
  </tbody></table>

## aclnnBinaryCrossEntropyWithLogitsTargetBackward

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 200px">
    <col style="width: 162px">
    <col style="width: 882px">
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
        <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize.</td>
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
  - **aclnnBinaryCrossEntropyWithLogitsTargetBackward** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_binary_cross_entropy_with_logits_target_backward.h"

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
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> targetShape = {2, 2};
  std::vector<int64_t> weightShape = {2, 2};
  std::vector<int64_t> posWeightShape = {2, 2};
  std::vector<int64_t> gradTargetShape = {2, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* posWeightDeviceAddr = nullptr;
  void* gradTargetDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* gradTarget = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* posWeight = nullptr;
  std::vector<float> gradOutputHostData = {0, 1, 2, 3};
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> targetHostData = {0.1, 0.1, 0.1, 0.1};
  std::vector<float> weightHostData = {0, 1, 2, 3};
  std::vector<float> posWeightHostData = {0, 1, 2, 3};
  std::vector<float> gradTargetHostData = {0, 0, 0, 0};
  int64_t reduction = 0;

  // Create a gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a posWeight aclTensor.
  ret = CreateAclTensor(posWeightHostData, posWeightShape, &posWeightDeviceAddr, aclDataType::ACL_FLOAT, &posWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gradTarget aclTensor.
  ret = CreateAclTensor(gradTargetHostData, gradTargetShape, &gradTargetDeviceAddr, aclDataType::ACL_FLOAT, &gradTarget);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBinaryCrossEntropyWithLogitsTargetBackward API call example
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  // Call the first-phase API of aclnnBinaryCrossEntropyWithLogitsTargetBackward.
  ret = aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize(gradOutput, self, target, weight, posWeight,
      reduction, gradTarget, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize failed. ERROR: %d\n",
                                          ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnBinaryCrossEntropyWithLogitsTargetBackward.
  ret = aclnnBinaryCrossEntropyWithLogitsTargetBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyWithLogitsTargetBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(gradTargetShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradTargetDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(weight);
  aclDestroyTensor(posWeight);
  aclDestroyTensor(gradTarget);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(posWeightDeviceAddr);
  aclrtFree(gradTargetDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
