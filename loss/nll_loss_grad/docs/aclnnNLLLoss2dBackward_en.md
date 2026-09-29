# aclnnNLLLoss2dBackward

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

Performs backpropagation of the negative log-likelihood loss.

## Prototype

  Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnNLLLoss2dBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnNLLLoss2dBackward** is called to perform computation.

```cpp
aclnnStatus aclnnNLLLoss2dBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    const aclTensor *target,
    const aclTensor *weight,
    int64_t          reduction,
    int64_t          ignoreIndex,
    aclTensor       *totalWeight,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnNLLLoss2dBackward(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnNLLLoss2dBackwardGetWorkspaceSize

- **Parameters**

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
        <td>The shape is three-dimensional (the first dimension is N) or one-dimensional (the number of elements is 1).</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>aclTensor to be input.</td>
        <td><ul><li>The shape is 4-dimensional. The first dimension is N, indicating the batch size. The second dimension is C, indicating the number of classes. </li><li>The shape of the 0th, 2nd, and 3rd dimensions of self must be the same as that of the 0th, 1st, and 2nd dimensions of target, respectively. Otherwise, false is returned.</li></ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>target (aclTensor*) </td>
        <td>Input</td>
        <td>Indicates the true label.</td>
        <td><ul><li>The shape of y in the formula is 3-dimensional.</li><li>The shape of the 0th, 1st, and 2nd dimensions of target is the same as that of the 0th, 2nd, and 3rd dimensions of self, respectively.</li><li>The value range of each element is [0, C - 1]</li></ul></td>.
        <td>INT64, UINT8, INT32</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight (aclTensor*) </td>
        <td>Input</td>
        <td>Indicates the scaling weight of each class.</td>
        <td>The shape of w in the formula is (C,).</td>
        <td>The data type is the same as that of self.</td>
        <td>ND</td>
        <td>(C,)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduction (int64_t) </td>
        <td>Input</td>
        <td>Specifies the reduction to be applied to the output.</td>
        <td><ul>The value can be 0 (none), 1 (mean), or 2 (sum). <li>'none' indicates that no reduction is applied.</li><li>'mean' indicates that the sum of the output is divided by the number of elements in the output.</li><li>'sum' indicates that the output is summed up.</li><li>When reduction is 0, the shape of target must be the same as that of gradOutput. Otherwise, false is returned.</li></ul></td>
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
        <td>-</td>
        <td>When reduction is set to mean, totalWeight is obtained by using the weight at the corresponding position of the target and removing the weight corresponding to ignoreIndex. The remaining weights are summed up. When reduction is set to other values, this parameter is not processed by default.</td>
        <td>The data type is the same as that of weight.</td>
        <td>ND</td>
        <td>(1,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>-</td>
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

- **Returns**

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
      <td>The input gradOutput, self, target, weight, totalWeight, or out is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The data type of gradOutput, self, target, weight, totalWeight, or out is not supported.</td>
      </tr>
      <tr>
      <td>The data types of gradOutput, self, weight, totalWeight, and out are inconsistent.</td>
      </tr>
      <tr>
      <td>The target is not a 3D tensor, and self is not a 4D tensor.</td>
      </tr>
       <tr>
      <td>The number of elements in weight is not C.</td>
      </tr>
      <tr>
      <td>The number of elements in dimensions 0, 2, and 3 of self is not equal to the number of elements in dimensions 0, 1, and 2 of target.</td>
      </tr>
      <tr>
      <td>The number of elements in totalWeight is not 1.</td>
      </tr>
      <tr>
      <td>When reduction is set to none, the number of dimensions of gradOutput is not 3, or the number of elements in dimensions 0, 1, and 2 of gradOutput is not equal to the number of elements in dimensions 0, 1, and 2 of target.</td>
      </tr>
      <tr>
      <td>When reduction is not none, the number of dimensions of gradOutput is greater than 1 or the number of elements is not 1.</td>
      </tr>
      <tr>
      <td>The value of reduction is not in the range of 0 to 2.</td>
      </tr>
      </tbody>
      </table>

## aclnnNLLLoss2dBackward

- **Parameters**

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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnNLLLoss2dBackwardGetWorkspaceSize.</td>
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

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
    - **aclnnNLLLoss2dBackward** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_nll_loss2d_backward.h"

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
  std::vector<int64_t> gradShape = {3, 1, 1};
  std::vector<int64_t> selfShape = {3, 5, 1, 1};
  std::vector<int64_t> targetShape = {3, 1, 1};
  std::vector<int64_t> weightShape = {5};
  std::vector<int64_t> totalWeightShape = {1};
  std::vector<int64_t> outShape = {3, 5, 1, 1};

  void* gradDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* totalWeightDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* grad = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* totalWeight = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> gradHostData = {2.7, 2.6, 2.5};
  std::vector<float> selfHostData = {4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5};
  std::vector<int64_t> targetHostData = {2, 3, 1};
  std::vector<float> weightHostData = {1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<float> totalWeightHostData = {1.0};
  std::vector<float> outHostData = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int64_t reduction = 0;
  int64_t ignoreIndex = -100;

  // Create a grad aclTensor.
  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT64, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a totalWeight aclTensor.
  ret = CreateAclTensor(totalWeightHostData, totalWeightShape, &totalWeightDeviceAddr,
                        aclDataType::ACL_FLOAT, &totalWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnNLLLoss2dBackward.
  ret = aclnnNLLLoss2dBackwardGetWorkspaceSize(grad, self, target, weight, reduction, ignoreIndex, totalWeight, out,
                                               &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLoss2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnNLLLoss2dBackward.
  ret = aclnnNLLLoss2dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLoss2dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device buffer to the host buffer. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(grad);
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(weight);
  aclDestroyTensor(totalWeight);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(gradDeviceAddr);
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
