# aclnnNLLLossBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/nll_loss_grad)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     √    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>   |     √    |

## Function

- Function: performs backpropagation of the negative log likelihood loss function.
- Formula:
  - reduction=mean:

    $$
    x\_grad_{target(t)} =\begin{cases}
    (-gradOutput * w_{(target(t))}) / totalweight &, target(t)=1 \\
    0 &, target(t)=0
    \end{cases}
    $$

  - reduction=sum:

    $$
    x\_grad_{target(t)} =\begin{cases}
    -gradOutput * w_{(target(t))} &, target(t)=1 \\
    0 &, target(t)=0
    \end{cases}
    $$

  - reduction=none:

    $$
    x\_grad_{target(t)} =\begin{cases}
    -gradOutput_t * w_{(target(t))} &, target(t)=1 \\
    0 &, target(t)=0
    \end{cases}
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnNLLLossBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnNLLLossBackward** is called to perform computation.

```cpp
aclnnStatus aclnnNLLLossBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    const aclTensor *target,
    const aclTensor *weight,
    int64_t          reduction,
    int64_t          ignoreIndex,
    const aclTensor *totalWeight,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnNLLLossBackward(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnNLLLossBackwardGetWorkspaceSize

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
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>gradOutput (aclTensor*)</td>
        <td>Input</td>
        <td>aclTensor to be input.</td>
        <td>The shape is (N), 1D (with one element), or (). <ul><li>When reduction is 0 and the shape of self is (N, C), the shape can be (N). </li><li>When reduction is 0 and the shape of self is (C), the shape can be 1D (with one element) or (). </li><li>When reduction is not 0, the shape can be 1D (with one element) or ().</li></ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>aclTensor to be input.</td>
        <td>The data type and shape are the same as those of the output parameter out. <ul><li>The shape is (N, C) or (C), where N indicates the batch size and C indicates the number of classes. </li><li>When the shape of target is (N), the shape of self must be (N, C). </li><li>When the shape of target is (), the shape of self must be (C).</li></ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>target (aclTensor*) </td>
        <td>Input</td>
        <td>Indicates the true label.</td>
        <td>The shape is (N) or (). The value range of each element is [0, C – 1].</td>
        <td>INT64, UINT8, INT32</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight (aclTensor*) </td>
        <td>Input</td>
        <td>Indicates the scaling weight of each class.</td>
        <td>w in the formula, whose shape is (C,).</td>
        <td>The data type is the same as that of self.</td>
        <td>ND</td>
        <td>(C)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduction (int64_t) </td>
        <td>Input</td>
        <td>Indicates the calculation method of the loss function.</td>
        <td><ul> 0(none)|1(mean)|2(sum) is supported. <li>'none' indicates that no reduction is applied.</li><li>'mean' indicates that the sum of the output will be divided by the number of elements in the output.</li><li>'sum' indicates that the output will be summed up.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ignoreIndex (int64_t) </td>
        <td>Input</td>
         <td>Specifies a target value that is ignored and does not affect the input gradient.
        </td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
          <tr>
        <td>totalWeight (aclTensor*) </td>
        <td>Input</td>
        <td>totalWeight in the formula.</td>
        <td>When reduction is set to mean, totalWeight is obtained by using the weight at the corresponding position of the target and removing the weight corresponding to ignoreIndex. The remaining weights are summed up. When reduction is set to other values, this parameter is not processed by default.</td>
        <td>The data type is the same as that of weight.</td>
        <td>ND</td>
        <td>(1,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>`out` in the formula.</td>
        <td>Its shape is the same as that of self.</td>
        <td>The data type is the same as that of self.</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Size of the workspace to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

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
      <td>The input gradOutput, self, target, weight, out, reduction, and totalWeight are null pointers.</td>
      </tr>
      <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of gradOutput, self, target, weight, and totalWeight are not supported.</td>
      </tr>
      <tr>
      <td>The data types of gradOutput, self, weight, and totalWeight are inconsistent.</td>
      </tr>
       <tr>
      <td>The shapes of gradOutput, self, weight, out, and totalWeight are incorrect.</td>
      </tr>
      <tr>
      <td>The value of reduction is not in the range of 0 to 2.</td>
      </tr>
      </tbody>
      </table>

## aclnnNLLLossBackward

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
      <td>Size of the workspace requested on the device, which is obtained by the first segment of aclnnNLLLossBackwardGetWorkspaceSize.</td>
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
    - **aclnnNLLLossBackward** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_nll_loss_backward.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> gradOutputShape = {2};
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> targetShape = {2};
  std::vector<int64_t> weightShape = {3};
  std::vector<int64_t> totalWeightShape = {1};
  std::vector<int64_t> outShape = {2, 3};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* totalWeightDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* totalWeight = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> gradOutputHostData = {3.1, 6.5};
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5};
  std::vector<int32_t> targetHostData = {0, 2};
  std::vector<float> weightHostData = {1.1, 1.2, 1.3};
  std::vector<float> totalWeightHostData = {0};
  std::vector<float> outHostData(6, 0);
  int64_t reduction = 0;
  int64_t ignoreIndex = -100;
  // Create a gradOutput aclTensor.
  ret =
      CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT32, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a totalWeight aclTensor.
  ret = CreateAclTensor(totalWeightHostData, totalWeightShape, &totalWeightDeviceAddr, aclDataType::ACL_FLOAT,
                        &totalWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnNLLLossBackward.
  ret = aclnnNLLLossBackwardGetWorkspaceSize(gradOutput, self, target, weight, reduction, ignoreIndex, totalWeight, out,
                                             &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLossBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnNLLLossBackward.
  ret = aclnnNLLLossBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLossBackward failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(totalWeight);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(totalWeightDeviceAddr);
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
