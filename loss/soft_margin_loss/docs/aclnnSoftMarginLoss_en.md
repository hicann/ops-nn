# aclnnSoftMarginLoss

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:40:04.859Z -->

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

- Description: Computes the binary classification logistic loss between input self and target target. reduction specifies the reduction to apply to the output and supports `none`, `mean`, and `sum`. `none` indicates no reduction is applied, `mean` indicates the sum of the output will be divided by the number of elements in the output, and `sum` indicates the output will be summed.

- Formula:

  When reduction is 'none':

  $$
  \text{out}(self,target)=L=log(1+\exp(-target[i]*self[i]))
  $$

  When reduction is not 'none':

$$
\text{out}(self,target)=
\begin{cases}
\operatorname{mean}(L),& \text{if reduction}= \text{'mean'}\\ \operatorname{sum}(L), & \text{if reduction} = \text{'sum'}
\end{cases}
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnSoftMarginLossGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnSoftMarginLoss" API to perform the computation.

```cpp
aclnnStatus aclnnSoftMarginLossGetWorkspaceSize(
  const aclTensor* self,
  const aclTensor* target,
  int64_t reduction,
  aclTensor* out,
  uint64_t* workspaceSize,
  aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnSoftMarginLoss(
  void* workspace,
  uint64_t workspaceSize,
  aclOpExecutor* executor,
  aclrtStream stream)
```

## aclnnSoftMarginLossGetWorkspaceSize

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
      <td class="tg-0pky">self (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor, the input `self` in the formula.</td>
      <td class="tg-0pky">The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with target.<br>The data type must comply with the data type deduction rule with target.</td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">target (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Ground truth labels, the input `target` in the formula.</td>
      <td class="tg-0pky">The shape must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast relationship</a> with self.<br>The data type must conform to the data type deduction rule with self.</td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">reduction (int64_t)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Specifies the reduction to be applied to the output, which is the parameter reduction in the formula.</td>
      <td class="tg-0pky">Supports 0('none')|1('mean')|2('sum').<br>-'none' indicates that no reduction is applied<br>-'mean' indicates that the sum of the output will be divided by the number of elements in the output<br>-'sum' indicates that the output will be summed</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">out (aclTensor*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">The loss function output out in the formula.</td>
      <td class="tg-0lax">If reduction is set to 'none', the shape must be the same as the shape of self and target after <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a>. Otherwise, out is a scalar.</td>
      <td class="tg-0lax">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">0-8 when reduction is 'none' (matching the broadcast shape); 0-dimensional scalar when reduction is 'mean' or 'sum'</td>
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
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the op executor, which contains the operator computing flow.</td>
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
      <th class="tg-0pky">Return value</th>
      <th class="tg-0pky">Error code</th>
      <th class="tg-0pky">Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">The self, target, or out pointer is null.</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="3">161002</td>
      <td class="tg-0pky">The data type of self, target, or out is not supported.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shapes of self and target do not comply with the <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> rules, or the shape after <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> is inconsistent with that of out.</td>
    </tr>
    <tr>
      <td class="tg-0pky">reduction is not one of 0('none'), 1('mean'), or 2('sum').</td>
    </tr>
  </tbody>
  </table>

## aclnnSoftMarginLoss

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
          <td>The workspace memory address applied for on the device.</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Input</td>
          <td>The workspace size applied for on the device, obtained by the first-phase API aclnnSoftMarginLossGetWorkspaceSize.</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>Input</td>
          <td>Executor that contains the operator computation flow.</td>
        </tr>
        <tr>
          <td>stream</td>
          <td>Input</td>
          <td>Stream where the task is executed.</td>
        </tr>
      </tbody>
    </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnSoftMarginLoss defaults to a deterministic implementation.
- Shape and rank: self/target must have a rank of at most 8 (0-dimensional scalars are also accepted).
- Empty tensors: empty inputs are supported.
- Data format: only ND is supported.
- Numerical boundaries (the operator computes in float32 precision, which also applies to FLOAT16/BFLOAT16 inputs):
  - Overflow: when an element satisfies $-target \times self > \ln(FLT\_MAX) \approx 88.72$, $\exp(-target \times self)$ overflows to +inf, and the output is $log(1+inf)=inf$; when reduction is 'mean' or 'sum', any element being inf makes the scalar output inf. A typical scenario is a severely wrong prediction direction (for example, target=1 and self takes a large negative value).
  - Underflow: the implementation uses a stable log1p form of $\log(1+\exp(-target \times self))$. For large $target \times self$, the output is approximately $\exp(-target \times self)$ instead of becoming 0 because $1+\exp(-target \times self)$ rounds to 1.

## Example

The sample code is as follows. It is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_soft_margin_loss.h"

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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API reference.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Custom construction is required based on the API.
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> targetShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0.3, 0.7, 0.5, 0.5};
  std::vector<float> targetHostData = {-1, 1, 1, -1};
  std::vector<float> outHostData = {0, 0, 0, 0};
  int64_t reduction = 0;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSoftMarginLoss.
  ret = aclnnSoftMarginLossGetWorkspaceSize(self, target, reduction, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftMarginLossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSoftMarginLoss.
  ret = aclnnSoftMarginLoss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftMarginLoss failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait synchronously for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value. Copy the result in device memory to the host. Modify the code according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the code according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(out);

 // 7. Release device resources. Modify the code according to the specific API definition.
 aclrtFree(selfDeviceAddr);
 aclrtFree(targetDeviceAddr);
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
