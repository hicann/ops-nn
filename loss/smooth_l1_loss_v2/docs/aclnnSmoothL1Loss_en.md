# aclnnSmoothL1Loss

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/smooth_l1_loss_v2)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     √    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>   |     √    |

## Function

- This API is used to calculate the SmoothL1 loss function.
- Formula:

  If `reduction` is `none`, the loss function for batch N is defined as follows:

  $$
  \ell(self,target) = L = \{l_1,\dots,l_N\}^\top
  $$

  $l_n$ is calculated as follows:

  $$
  l_n = \begin{cases}
  0.5(self_n-target_n)^2/beta, & if |self_n-target_n| < beta \\
  |self_n-target_n| - 0.5*beta, &  otherwise
  \end{cases}
  $$

  If `reduction` is `mean` or `sum`:

  $$
  \ell(self,target)=\begin{cases}
  mean(L), & \text{if reduction} = \text{mean}\\
  sum(L), & \text{if reduction} = \text{sum}
  \end{cases}
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnSmoothL1LossGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnSmoothL1Loss** is called to perform computation.

```cpp
aclnnStatus aclnnSmoothL1LossGetWorkspaceSize(
      const aclTensor* self, 
      const aclTensor* target, 
      int64_t reduction, 
      float beta, 
      aclTensor* result, 
      uint64_t* workspaceSize, 
      aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnSmoothL1Loss(
    void* workspace, 
    uint64_t workspaceSize,  
    aclOpExecutor* executor, 
    aclrtStream stream)
```

## aclnnSmoothL1LossGetWorkspaceSize

- **Parameters:**

  <table class="tg" style="undefined;table-layout: fixed; width: 1582px"><colgroup>
  <col style="width: 217px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 350px">
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 120px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor, which is the input self in the formula.</td>
      <td class="tg-0pky">The shape must be broadcastable with that of target (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>).<br>Its data type and the data type of target must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND, NCL, NCHW, NHWC</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">target (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Real label, which is the input y in the formula.</td>
      <td class="tg-0pky">The shape must be broadcastable with that of self (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>).<br>Its data type and the data type of self must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND, NCL, NCHW, NHWC</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">reduction (int64_t) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Reduction to be applied to the output, which is the parameter reduction in the formula.</td>
      <td class="tg-0pky">The value can be 0 (none), 1 (mean), or 2 (sum).<br>'none' indicates that no reduction is applied.<br>'mean' indicates that the sum of the output will be divided by the number of elements in the output.<br>'sum' indicates that the output will be summed up. </td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">beta (float) </td>
      <td class="tg-0lax">Enter </td>.
      <td class="tg-0lax">Computational attribute, specifying the value to change between L1 and L2 losses.</td>
      <td class="tg-0lax">The value must be non-negative.</td>
      <td class="tg-0lax">FLOAT</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">result (aclTensor*) </td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">The loss function ℓ(self, target) in the formula.</td>
      <td class="tg-0lax">When reduction is none, the shape is the same as the result of the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> between self and target. When reduction is mean or sum, the shape is [1], and the data type is a data type that can be converted after self and target are inferred. (For details, see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>.)</td>
      <td class="tg-0lax">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0lax">ND, NCL, NCHW, NHWC</td>
      <td class="tg-0lax">1-8</td>
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
      <td class="tg-0pky">executor (aclOpExecutor**) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the operator executor, including the operator execution process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **Returns:**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table class="tg" style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 269px">
  <col style="width: 135px">
  <col style="width: 746px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Return Value</th>
      <th class="tg-0pky">Error Code</th>
      <th class="tg-0pky">Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">The input self, target, or result is a null pointer.</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="7">161002</td>
      <td class="tg-0pky">The data type of self, target, or result is not supported.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shape of self, target, or result does not meet the constraints.</td>
    </tr>
    <tr>
      <td class="tg-0pky">reduction does not meet the constraints.</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta does not meet the constraints.</td>
    </tr>
    <tr>
      <td class="tg-0lax">The data types of self and target do not meet the data type inference rules.</td>
    </tr>
    <tr>
      <td class="tg-0lax">The shapes of self and target do not meet the broadcast relationship.</td>
    </tr>
    <tr>
      <td class="tg-0lax">When reduction is set to 0, the shape of the result is inconsistent with the broadcast shape of self and target.</td>
    </tr>
  </tbody>
  </table>

## aclnnSmoothL1Loss

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
          <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnSmoothL1LossGetWorkspaceSize.</td>
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
    - **aclnnSmoothL1Loss** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_smooth_l1_loss.h"

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
  std::vector<int64_t> selfShape = {2, 2, 7, 7};
  std::vector<int64_t> targetShape = {2, 2, 7, 7};
  std::vector<int64_t> resultShape = {2, 2, 7, 7};

  // Create a self aclTensor.
  std::vector<float> selfData(GetShapeSize(selfShape)* 2, 1);
  aclTensor* self = nullptr;
  void *selfDeviceAddr = nullptr;
  ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a target aclTensor.
  std::vector<float> targetData(GetShapeSize(targetShape)* 2, 1);
  aclTensor* target = nullptr;
  void *targetDeviceAddr = nullptr;
  ret = CreateAclTensor(targetData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT16, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a result aclTensor.
  std::vector<float> resultData(GetShapeSize(resultShape)* 2, 1);
  aclTensor* result = nullptr;
  void *resultDeviceAddr = nullptr;
  ret = CreateAclTensor(resultData, resultShape, &resultDeviceAddr, aclDataType::ACL_FLOAT16, &result);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSmoothL1Loss.
  int64_t reduction = 0;
  float beta = 1.0;
  ret = aclnnSmoothL1LossGetWorkspaceSize(self, target, reduction, beta, result, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSmoothL1LossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSmoothL1Loss.
  ret = aclnnSmoothL1Loss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSmoothL1Loss failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(resultShape);
  std::vector<float> resultOutData(size, 0);
  ret = aclrtMemcpy(resultOutData.data(), resultOutData.size() * sizeof(resultOutData[0]), resultDeviceAddr,
                    size * sizeof(resultOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultOutData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(result);
  // 7. Release device resources. Set the parameters based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(resultDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
