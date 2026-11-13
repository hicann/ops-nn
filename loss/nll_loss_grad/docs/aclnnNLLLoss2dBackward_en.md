# aclnnNLLLoss2dBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:34:00.471Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

Performs the backward propagation of the negative log-likelihood loss.

## Function Prototype

  Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnNLLLoss2dBackwardGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the computation process of the operator, and then call the "aclnnNLLLoss2dBackward" API to perform the computation.

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
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>gradOutput (aclTensor*)</td>
        <td>Input</td>
        <td>Input aclTensor.</td>
        <td>The shape is three-dimensional (the first dimension is N) or one-dimensional (with only one element).</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>Input aclTensor.</td>
        <td><ul><li>The shape is 4D. The first dimension N indicates the batch size, and the second dimension C indicates the number of classes.</li><li>The shape of the 0th, 2nd, and 3rd dimensions of self must be consistent with that of the 0th, 1st, and 2nd dimensions of target; otherwise, false is returned.</li></ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>target (aclTensor*)</td>
        <td>Input</td>
        <td>Represents the ground-truth labels.</td>
        <td><ul><li>y in the formula, with a 3D shape.</li><li>The shape of the 0th, 1st, and 2nd dimensions of target must be consistent with that of the 0th, 2nd, and 3rd dimensions of self.</li><li>The value range of each element is [0, C - 1].</li></ul></td>
        <td>INT64, UINT8, INT32</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight (aclTensor*)</td>
        <td>Input</td>
        <td>Represents the scaling weight for each class.</td>
        <td>w in the formula, with shape (C,).</td>
        <td>The data type must be consistent with self.</td>
        <td>ND</td>
        <td>(C,)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduction(int64_t)</td>
        <td>Input</td>
        <td>Specifies the reduction to apply to the output.</td>
        <td><ul>Supported values: 0(none)|1(mean)|2(sum).<li>'none' means no reduction is applied</li><li>'mean' means the sum of the output is divided by the number of elements in the output</li><li>'sum' means the output is summed</li><li>When reduction is 0, target and gradOutput must have consistent shapes. Otherwise, false is returned.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ignoreIndex(int64_t)</td>
        <td>Input</td>
         <td>Specifies a target value to be ignored. The value does not affect the input gradient.
        </td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
          <tr>
        <td>totalWeight (aclTensor*)</td>
        <td>Input</td>
        <td>-</td>
        <td>totalWeight is an input. When reduction is mean, the caller must pass in the sum of weights calculated based on target, weight, and ignoreIndex; when reduction is set to other values, this parameter is not involved in the computation.</td>
        <td>Data type is the same as that of weight.</td>
        <td>ND</td>
        <td>(1,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>-</td>
        <td>Shape is the same as that of self.</td>
        <td>Data type is consistent with that of self.</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      </tbody>
      </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 276px">
      <col style="width: 132px">
      <col style="width: 836px">
      </colgroup>
      <thead>
      <tr>
      <th>Return value</th>
      <th>Error code</th>
      <th>Description</th>
      </tr></thead>
      <tbody>
      <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>gradOutput, self, target, weight, totalWeight, or out is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td> The data type of gradOutput, self, target, weight, totalWeight, or out is not within the supported range.</td>
      </tr>
      <tr>
      <td> The data types of gradOutput, self, weight, totalWeight, and out are inconsistent.</td>
      </tr>
      <tr>
      <td>target is not a 3D tensor, or self is not a 4D tensor.</td>
      </tr>
       <tr>
      <td>The number of elements in weight is not C.</td>
      </tr>
      <tr>
      <td>The number of elements in dimensions 0, 2, and 3 of self is not equal to that in dimensions 0, 1, and 2 of target.</td>
      </tr>
      <tr>
      <td> The number of elements in totalWeight is not 1.</td>
      </tr>
      <tr>
      <td>When reduction is none, the number of dimensions of gradOutput is not 3, or the number of elements in dimensions 0, 1, and 2 of gradOutput is not equal to that in dimensions 0, 1, and 2 of target.</td>
      </tr>
      <tr>
      <td>When reduction is not none, the number of dimensions of gradOutput is greater than 1 or the number of elements is not 1.</td>
      </tr>
      <tr>
      <td>The value of reduction is not in the range 0-2.</td>
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
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address requested on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the device, obtained by calling the first-phase API aclnnNLLLoss2dBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the operator. It contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
    </tbody>
     </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
    - aclnnNLLLoss2dBackward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  // Boilerplate for resource initialization.
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
  // Call aclrtMalloc to allocate device-side memory
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API reference.
  // Fill in deviceId based on the actual device in use
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs as required by the API
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

  // Create the grad aclTensor
  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the target aclTensor
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT64, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the totalWeight aclTensor.
  ret = CreateAclTensor(totalWeightHostData, totalWeightShape, &totalWeightDeviceAddr,
                        aclDataType::ACL_FLOAT, &totalWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnNLLLoss2dBackward.
  ret = aclnnNLLLoss2dBackwardGetWorkspaceSize(grad, self, target, weight, reduction, ignoreIndex, totalWeight, out,
                                               &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLoss2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // Request device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnNLLLoss2dBackward.
  ret = aclnnNLLLoss2dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLoss2dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it according to the definition of the specific API.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify it according to the definition of the specific API.
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
