# aclnnSoftMarginLossBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:41:03.623Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Not Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

Computes the backward propagation of the binary classification logistic loss function of [aclnnSoftMarginLoss](../../soft_margin_loss/docs/aclnnSoftMarginLoss.md). The reduction parameter specifies the calculation method of the loss function, supporting 'none', 'mean', and 'sum'. 'none' indicates that no reduction is applied, 'mean' indicates that the sum of the output will be divided by the number of elements in the output, and 'sum' indicates that the output will be summed.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnSoftMarginLossBackwardGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the computation flow of the operator, and then call the "aclnnSoftMarginLossBackward" API to perform the computation.

```Cpp
aclnnStatus aclnnSoftMarginLossBackwardGetWorkspaceSize(
    const aclTensor* gradOutput,
    const aclTensor* self,
    const aclTensor* target,
    int64_t          reduction,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnSoftMarginLossBackward(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnSoftMarginLossBackwardGetWorkspaceSize

- **Parameters**

    <table class="tg" style="undefined;table-layout: fixed; width: 1547px"><colgroup>
    <col style="width: 217px">
    <col style="width: 120px">
    <col style="width: 280px">
    <col style="width: 350px">
    <col style="width: 200px">
    <col style="width: 115px">
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
        <td class="tg-0pky">gradOutput (aclTensor*)</td>
        <td class="tg-0pky">Input</td>
        <td class="tg-0pky">Input gradient for backward propagation.</td>
        <td class="tg-0pky">shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self and target.<br>The data type must satisfy the data type derivation rule with the data types of self and target (see also <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</td>
        <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
        <td class="tg-0pky">ND</td>
        <td class="tg-0pky">1-8</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">self (aclTensor*)</td>
        <td class="tg-0pky">Input</td>
        <td class="tg-0pky">Input tensor.</td>
        <td class="tg-0pky">The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with gradOutput and target.<br>The data type and the data types of gradOutput and target must satisfy the data type derivation rule (see also <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">mutual conversion relationship</a>).</td>
        <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
        <td class="tg-0pky">ND</td>
        <td class="tg-0pky">1-8</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">target (aclTensor*)</td>
        <td class="tg-0pky">Input</td>
        <td class="tg-0pky">Actual label, corresponding to input y in the formula.</td>
        <td class="tg-0pky">The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with gradOutput and self.<br>The data type and the data types of gradOutput and target must satisfy the data type derivation rule (see also <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">mutual conversion relationship</a>).</td>
        <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
        <td class="tg-0pky">ND</td>
        <td class="tg-0pky">1-8</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">reduction (int64_t)</td>
        <td class="tg-0pky">Input</td>
        <td class="tg-0pky">Specifies the reduction to apply to the output, which is the parameter reduction in the formula.</td>
        <td class="tg-0pky">Supports 0(none)|1(mean)|2(sum).<br>'none' indicates that no reduction is applied<br>'mean' indicates that the sum of the output will be divided by the number of elements in the output<br>'sum' indicates that the output will be summed</td>
        <td class="tg-0pky">INT64</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">out (aclTensor*)</td>
        <td class="tg-0pky">Output</td>
        <td class="tg-0pky">Computed output.</td>
        <td class="tg-0pky">The shape is the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> result of gradOutput, self, and target.</td>
        <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
        <td class="tg-0pky">ND</td>
        <td class="tg-0pky">1-8</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">workspaceSize (uint64_t*)</td>
        <td class="tg-0pky">Output</td>
        <td class="tg-0pky">Returns the workspace size to be allocated on the Device side.</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
      </tr>
      <tr>
        <td class="tg-0pky">executor (aclOpExecutor**)</td>
        <td class="tg-0pky">Output</td>
        <td class="tg-0pky">Returns the op executor, which contains the operator computation flow.</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
      </tr>
    </tbody></table>

- **Return value**:

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
        <td class="tg-0pky">The passed gradOutput, self, target, or out is a null pointer.</td>
      </tr>
      <tr>
        <td class="tg-0pky" rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td class="tg-0pky" rowspan="3">161002</td>
        <td class="tg-0pky">The data type of gradOutput, self, target, or out is not within the supported range.</td>
      </tr>
      <tr>
        <td class="tg-0lax">The shapes of gradOutput, self and target, and out do not satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> rule.</td>
      </tr>
      <tr>
        <td class="tg-0pky">The shape of gradOutput, self, and target after <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> is inconsistent with that of out.</td>
      </tr>
    </tbody>
    </table>

## aclnnSoftMarginLossBackward

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
      <col style="width: 200px">
      <col style="width: 150px">
      <col style="width: 800px">
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
          <td>Workspace memory address allocated on the Device side.</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Input</td>
          <td>Workspace size allocated on the Device side, obtained by calling the first-phase API aclnnSoftMarginLossBackwardGetWorkspaceSize.</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>Input</td>
          <td>Executor of the op, containing the computation flow of the operator.</td>
        </tr>
        <tr>
          <td>stream</td>
          <td>Input</td>
          <td>Stream that executes the task.</td>
        </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Dynamic shapes and dynamic ranks are supported (DynamicShapeSupportFlag=true, DynamicRankSupportFlag=true); input shapes do not need to be fixed at compile time.
- Empty tensors: after parameter, broadcast, and output shape validation, if the output has zero elements, the ACLNN interface returns success with zero workspace, without performing broadcast copies or executing the numerical kernel.
- Deterministic computation:
  - aclnnSoftMarginLossBackward defaults to a deterministic implementation.

## Example

Sample code is as follows for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_soft_margin_loss_backward.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            aclFinalize();
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host side to the device side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            aclrtFree(*deviceAddr);
            *deviceAddr = nullptr;
            return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) device/stream initialization. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* out = nullptr;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int64_t reduction = 1;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. gradOutput must be custom-constructed based on the API.
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> targetShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  auto size = GetShapeSize(outShape);
  std::vector<float> gradOutputHostData = {0, 1, 2, 3};
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> targetHostData = {1, 1, 1, 1};
  std::vector<float> outHostData(4, 0);
  std::vector<float> resultData(size, 0);
  // Create the gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr,
                        aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, goto Cleanup);
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, goto Cleanup);
  // Create the target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, goto Cleanup);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, goto Cleanup);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  // Call the first-phase API aclnnSoftMarginLossBackward.
  ret = aclnnSoftMarginLossBackwardGetWorkspaceSize(gradOutput, self, target, reduction, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftMarginLossBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
            goto Cleanup);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); goto Cleanup);
  }
  // Call the second-phase API aclnnSoftMarginLossBackward.
  ret = aclnnSoftMarginLossBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftMarginLossBackward failed. ERROR: %d\n", ret); goto Cleanup);

  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); goto Cleanup);

  // 5. Obtain the output value and copy the result from the device memory to the host memory. Modify the code based on the API definition.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); goto Cleanup);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

Cleanup:
  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  if (gradOutput != nullptr) {
    aclDestroyTensor(gradOutput);
  }
  if (self != nullptr) {
    aclDestroyTensor(self);
  }
  if (target != nullptr) {
    aclDestroyTensor(target);
  }
  if (out != nullptr) {
    aclDestroyTensor(out);
  }

  // 7. Release device resources. Modify the code based on the API definition.
  if (gradOutputDeviceAddr != nullptr) {
    aclrtFree(gradOutputDeviceAddr);
  }
  if (selfDeviceAddr != nullptr) {
    aclrtFree(selfDeviceAddr);
  }
  if (targetDeviceAddr != nullptr) {
    aclrtFree(targetDeviceAddr);
  }
  if (outDeviceAddr != nullptr) {
    aclrtFree(outDeviceAddr);
  }
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  if (stream != nullptr) {
    aclrtDestroyStream(stream);
  }
  aclrtResetDevice(deviceId);
  aclFinalize();
  return ret;
}
```

### Rank Constraints

The ACLNN interface checks that the original rank of each of gradOutput, self, and target is no greater than 8. The effective-rank compatibility rules for kernel/GEIR do not extend the original-rank limit of this interface.
