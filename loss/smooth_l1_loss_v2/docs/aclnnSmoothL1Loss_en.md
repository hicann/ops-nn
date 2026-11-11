# aclnnSmoothL1Loss

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:38:56.955Z -->

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Computes the SmoothL1 loss function.
- Formula:

  For a batch of size N, when `reduction` is none, the loss function is defined as:

  $$
  \ell(self,target) = L = \{l_1,\dots,l_N\}^\top
  $$

  where $l_n$ is:

  $$
  l_n = \begin{cases}
  0.5(self_n-target_n)^2/beta, & if |self_n-target_n| < beta \\
  |self_n-target_n| - 0.5*beta, &  otherwise
  \end{cases}
  $$

  When `reduction` is `mean` or `sum`,

  $$
  \ell(self,target)=\begin{cases}
  mean(L), & \text{if reduction} = \text{mean}\\
  sum(L), & \text{if reduction} = \text{sum}
  \end{cases}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must call "aclnnSmoothL1LossGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call "aclnnSmoothL1Loss" to execute the computation.

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
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor, which is input self in the formula.</td>
      <td class="tg-0pky">The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with target.<br>The data type must satisfy the data type derivation rule with target (see also <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND, NCL, NCHW, NHWC</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">target (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Ground truth label, which is input y in the formula.</td>
      <td class="tg-0pky">The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.<br>The data type must comply with the data type derivation rule with self (see also <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND, NCL, NCHW, NHWC</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">reduction (int64_t)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Specifies the reduction applied to the output, corresponding to the parameter reduction in the formula.</td>
      <td class="tg-0pky">Supports 0(none)|1(mean)|2(sum).<br>'none': no reduction is applied<br>'mean': the sum of the output is divided by the number of elements in the output<br>'sum': the output is summed</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">beta (float)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Computation attribute, specifying the value at which the loss changes between L1 and L2.</td>
      <td class="tg-0lax">The value must be non-negative.</td>
      <td class="tg-0lax">FLOAT</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">result (aclTensor*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">The loss function ℓ(self,target) output in the formula.</td>
      <td class="tg-0lax">When reduction is none, the shape is the same as the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> result of self and target. When reduction is mean or sum, the shape is [1]. The data type is the data type that self and target can be converted to after derivation (see also <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>). When reduction is mean or sum, the format of result must be ND. When result is of the DOUBLE type, the output is the DOUBLE value obtained by casting (<a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>) the computation result.</td>
      <td class="tg-0lax">FLOAT, FLOAT16, BFLOAT16, DOUBLE</td>
      <td class="tg-0lax">ND, NCL, NCHW, NHWC</td>
      <td class="tg-0lax">0-8</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size to be requested on the Device side.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor(aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor, which contains the operator computation process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

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
      <td class="tg-0pky">self, target, or result is a null pointer.</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="7">161002</td>
      <td class="tg-0pky">The data type of self, target, or result is not within the supported range.</td>
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
      <td class="tg-0lax">The data types of self and target do not satisfy the data type derivation rule.</td>
    </tr>
    <tr>
      <td class="tg-0lax">The shapes of self and target do not satisfy the broadcast relationship.</td>
    </tr>
    <tr>
      <td class="tg-0lax">When reduction=0, the shape of result is inconsistent with the broadcast shape of self and target.</td>
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
          <th>Parameter</th>
          <th>Input/Output</th>
          <th>Description</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>workspace</td>
          <td>Input</td>
          <td>Memory address of the workspace applied for on the Device.</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Input</td>
          <td>Size of the workspace applied for on the Device. The value is obtained by the first-phase API aclnnSmoothL1LossGetWorkspaceSize.</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>Input</td>
          <td>Executor that contains the operator computation flow.</td>
        </tr>
        <tr>
          <td>stream</td>
          <td>Input</td>
          <td>Stream for executing tasks.</td>
        </tr>
      </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
    - aclnnSmoothL1Loss defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  // Boilerplate, resource initialization
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
  // Call aclrtMalloc to allocate memory on the device
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to the device memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation
  // Set deviceId based on the actual device
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API
  std::vector<int64_t> selfShape = {2, 2, 7, 7};
  std::vector<int64_t> targetShape = {2, 2, 7, 7};
  std::vector<int64_t> resultShape = {2, 2, 7, 7};

  // Create the self aclTensor
  std::vector<float> selfData(GetShapeSize(selfShape)* 2, 1);
  aclTensor* self = nullptr;
  void *selfDeviceAddr = nullptr;
  ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the target aclTensor
  std::vector<float> targetData(GetShapeSize(targetShape)* 2, 1);
  aclTensor* target = nullptr;
  void *targetDeviceAddr = nullptr;
  ret = CreateAclTensor(targetData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the result aclTensor
  std::vector<float> resultData(GetShapeSize(resultShape)* 2, 1);
  aclTensor* result = nullptr;
  void *resultDeviceAddr = nullptr;
  ret = CreateAclTensor(resultData, resultShape, &resultDeviceAddr, aclDataType::ACL_FLOAT, &result);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSmoothL1Loss
  int64_t reduction = 0;
  float beta = 1.0;
  ret = aclnnSmoothL1LossGetWorkspaceSize(self, target, reduction, beta, result, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSmoothL1LossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSmoothL1Loss
  ret = aclnnSmoothL1Loss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSmoothL1Loss failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Wait for the task execution to complete
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device-side memory to the host side, which needs to be modified based on the definition of the specific API
  auto size = GetShapeSize(resultShape);
  std::vector<float> resultOutData(size, 0);
  ret = aclrtMemcpy(resultOutData.data(), resultOutData.size() * sizeof(resultOutData[0]), resultDeviceAddr,
                    size * sizeof(resultOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultOutData[i]);
  }

  // 6. Release aclTensor and aclScalar, which needs to be modified based on the definition of the specific API
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(result);
  // 7. Release device resources, which needs to be based on the parameters of the specific API definition
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
