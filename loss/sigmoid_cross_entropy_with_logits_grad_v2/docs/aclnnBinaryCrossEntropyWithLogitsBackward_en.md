# aclnnBinaryCrossEntropyWithLogitsBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/sigmoid_cross_entropy_with_logits_grad_v2)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     x    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>   |     √    |

## Function

Computes `logits` computation of the input `self`, and performs backward propagation of `BCELoss` using the computed logits and label value `target`.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnBinaryCrossEntropyWithLogitsBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    const aclTensor *target,
    const aclTensor *weightOptional,
    const aclTensor *posWeightOptional,
    int64_t          reduction,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnBinaryCrossEntropyWithLogitsBackward(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
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
        <td>gradOutput (aclTensor*) </td>
        <td>Input</td>
        <td>Gradient value of the previous step of network backpropagation.</td>
        <td>The shape can be broadcast to the shape of self. For details, see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a>.</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>self (aclTensor*) </td>
        <td>Input</td>
        <td>Computation result of the previous layer in the forward propagation of the network.</td>
        <td>-</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>target (aclTensor*) </td>
        <td>Input</td>
        <td>Label value of a sample.</td>
        <td>The value ranges from 0 to 1.</td>
        <td>Same as self</td>
        <td>ND</td>
        <td>Same as self</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightOptional (aclTensor*) </td>
        <td>Input</td>
        <td>Weight of binary cross entropy.</td>
        <td><ul><li>shape can be <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> to self shape. </li><li> When weightOptional is empty, an all-1 tensor is created based on the self shape.</li></ul></td>
        <td>The value must be the same as that of self.</td>;
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>posWeightOptional (aclTensor*) </td>
        <td>Input</td>
        <td>Positive class weight.</td>
        <td><ul><li>shape can be <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> to self shape. </li><li> When posWeightOptional is empty, an all-1 tensor is created based on the self shape.</li></ul></td>
        <td>The value must be the same as that of self.</td>;
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduction (int64_t) </td>
        <td>Input</td>
        <td>Reduction operation performed on the result of the backward gradient computation of binary cross entropy.</td>
        <td> supports 0(none)|1(mean)|2(sum). <ul><li>0 indicates that no operation is performed. </li><li>1 indicates the average value of the result. </li><li>2: indicates that the results are summed up.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*) </td>
        <td>Output</td>
        <td>Gradient computation result.</td>
        <td><a href="../../../docs/en/context/data_format.md">data format</a> must be consistent with self.</td>
        <td>Same as self</td>
        <td>-</td>
        <td>Same as self</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*) </td>
        <td>Output</td>
        <td>Size of the workspace required to be allocated on the device.</td>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
    </colgroup>
    <thead>
      <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed gradOutput, self, target, or out is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type or format of gradOutput, self, target, or out is not supported.</td>
      </tr>
      <tr>
      <td>When weightOptional and posWeightOptional are not null pointers, their data types or formats are not supported.</td>
      </tr>
      <tr>
      <td>The shapes of self, target, and out are inconsistent.</td>
      </tr>
      <tr>
      <td>If weightOptional and posWeightOptional are not null pointers, the shape of the two parameters cannot be <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcastable</a> to that of `self`.</td>
      </tr>
      <tr>
      <td>gradOutput cannot be <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcastable</a> to that of `self`.</td>
      </tr>
      <tr>
      <td>The value of reduction is not 0, 1, or 2.</td>
      </tr>
    </tbody>
    </table>

## aclnnBinaryCrossEntropyWithLogitsBackward

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
          <td>Address of the workspace to be allocated on the device.</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Input</td>
          <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API **aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize**.</td>
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
  - **aclnnBinaryCrossEntropyWithLogitsBackward** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_binary_cross_entropy_with_logits_backward.h"

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
  std::vector<int64_t> outShape = {2, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* posWeightDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* out = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* posWeight = nullptr;
  std::vector<float> gradOutputHostData = {0, 1, 2, 3};
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> targetHostData = {0.1, 0.1, 0.1, 0.1};
  std::vector<float> weightHostData = {0, 1, 2, 3};
  std::vector<float> posWeightHostData = {0, 1, 2, 3};
  std::vector<float> outHostData = {0, 0, 0, 0};
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
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBinaryCrossEntropyWithLogitsBackward API call example
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  // Call the first-phase API of aclnnBinaryCrossEntropyWithLogitsBackward.
  ret = aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize(gradOutput, self, target, weight, posWeight,
      reduction, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize failed. ERROR: %d\n",
                                          ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnBinaryCrossEntropyWithLogitsBackward.
  ret = aclnnBinaryCrossEntropyWithLogitsBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyWithLogitsBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
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
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(posWeightDeviceAddr);
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
