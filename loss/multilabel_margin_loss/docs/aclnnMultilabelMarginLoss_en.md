# aclnnMultilabelMarginLoss

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:31:18.018Z -->

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

- Description: Computes the multilabel margin loss.
- Formula:
self is the input, with shape (N,C) or (C), where N is the batch size and C is the number of classes. target indicates the ground-truth labels, with shape (N,C) or (C). The value of each element ranges from [-1, C - 1]. -1 is used for padding to ensure the same shape as the input; that is, the labels before the first -1 represent the ground-truth label yTrue of a sample. For example, if y=[0,3,-1,1], the ground-truth label yTrue is [0,3]. The formula for computing each sample is as follows:

  $$
    istarget[k]=\begin{cases}
      \ 1, &
      \text{k in yTrue}\\
      \ 0, &
      \text{otherwise}
  \end{cases}
  $$

  $$
    l_n=\sum^C_{j,istarget[j]=1}\sum^C_{i,istarget[i]=0} \frac{max(0,1-x[j]+x[i])}{C}
  $$

  When `reduction` is `none`,

  $$
  \ell(x, y) = L = \{l_1,\dots,l_N\}^\top
  $$

  If `reduction` is not `none`, then

  $$
  \ell(x, y) = \begin{cases}
      \sum_{n=1}^N \frac{1}{N} l_n, &
      \text{if reduction} = \text{mean;}\\
      \sum_{n=1}^N l_n, &
      \text{if reduction} = \text{sum.}
  \end{cases}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMultilabelMarginLossGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnMultilabelMarginLoss" API to perform the computation.

```Cpp
aclnnStatus aclnnMultilabelMarginLossGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* target,
    int64_t          reduction,
    aclTensor*       out,
    aclTensor*       isTarget,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnMultilabelMarginLoss(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnMultilabelMarginLossGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1475px"><colgroup>
  <col style="width: 205px">
  <col style="width: 120px">
  <col style="width: 320px">
  <col style="width: 320px">
  <col style="width: 130px">
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
      <td class="tg-0pky">Input tensor, input x in the formula.</td>
      <td class="tg-0pky">The shape is (N,C) or (C), where N indicates batch size and C indicates the number of classes. <br> 507034 Vector Core runtime timeout may occur when the number of elements in `self` is greater than 15000 * 20000.</td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1, 2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">target (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Ground-truth labels, input y in the formula.</td>
      <td class="tg-0pky">Shape is (N, C) or (C). The value range of each element is [-1, C - 1], padded with -1. Labels before the first -1 represent the real labels of samples.</td>
      <td class="tg-0pky">INT32, INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1, 2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">reduction (int64_t)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Specifies the reduction to be applied to the output, that is, the parameter reduction in the formula.</td>
      <td class="tg-0pky">Supports 0(none)|1(mean)|2(sum).<br>'none': no reduction is applied<br>'mean': the sum of the output is divided by the number of elements in the output<br>'sum': the output is summed</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">out (aclTensor*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Output loss, which is ℓ(x,y) in the formula.</td>
      <td class="tg-0lax">Shape (N) or ()</td>
      <td class="tg-0lax">Same as self</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">0, 1</td>
      <td class="tg-0lax"></td>
    </tr>
    <tr>
      <td class="tg-0pky">isTarget (aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Output `istarget` in the formula (0/1 mask derived from target).</td>
      <td class="tg-0pky">Shape (N, C) or (C)</td>
      <td class="tg-0pky">Same as self (FLOAT, FLOAT16, BFLOAT16)</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1, 2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size to be allocated on the Device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**)</td>
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

  aclnnStatus: return status code, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table class="tg" style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 269px">
  <col style="width: 120px">
  <col style="width: 761px">
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
      <td class="tg-0pky">The input self, target, out, and isTarget are null pointers.</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="6">161002</td>
      <td class="tg-0pky">The data type of self is not in the supported range (only FLOAT/FLOAT16/BFLOAT16 is supported).</td>
    </tr>
    <tr>
      <td class="tg-0pky">The data type of out or isTarget is inconsistent with that of self (the aclnn path checks out==self and isTarget==self, and the three must be of the same floating-point type).</td>
    </tr>
    <tr>
      <td class="tg-0pky">The data type of target is not in the supported range.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shapes of self, target, and isTarget do not meet the requirements in Parameters.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shape of out does not meet the requirements in Parameters.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The value of reduction is not within the range of 0-2.</td>
    </tr>
  </tbody>
  </table>

## aclnnMultilabelMarginLoss

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
          <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnMultilabelMarginLossGetWorkspaceSize.</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>Input</td>
          <td>Op executor that contains the operator computation process.</td>
        </tr>
        <tr>
          <td>stream</td>
          <td>Input</td>
          <td>Specifies the Stream for executing tasks.</td>
        </tr>
      </tbody>
    </table>

- **Return value**

  aclnnStatus: Return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
    - aclnnMultilabelMarginLoss defaults to a non-deterministic implementation.
- The data type of isTarget follows that of self (0/1 mask, torch contract is_target==self). self, out, and isTarget must be consistent (FLOAT/FLOAT16/BFLOAT16).
- Empty tensor: Only the N (batch) axis can be empty, that is, `[0,C]` (N=0, C≥1); a class dimension of C=0 (`[N,0]`/`[0,0]`/1D `[0]`) is invalid input and returns ACLNN_ERR_PARAM_INVALID(161002).
  For `[0,C]`, isTarget is an empty tensor; the value of out follows the general convention for reduction: out is an empty tensor when `reduction=none`, out is 0 when `reduction=sum`, and out is nan (0/0) when `reduction=mean`.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_multilabel_margin_loss.h"

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
 // Boilerplate: resource initialization.
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
  // Call aclrtMalloc to request memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of a contiguous tensor.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId according to the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction according to the API.
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> targetShape = {2, 3};
  std::vector<int64_t> outShape = {2};
  std::vector<int64_t> istargetShape = {2, 3};
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* istargetDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* out = nullptr;
  aclTensor* istarget = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5};
  std::vector<int32_t> targetHostData = {0, 1, 2, 3, 4, 5};
  std::vector<float> outHostData(2, 0);
  std::vector<float> istargetHostData(6, 0);
  int64_t reduction = 0;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT32, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the istarget aclTensor.
  ret = CreateAclTensor(istargetHostData, istargetShape, &istargetDeviceAddr, aclDataType::ACL_FLOAT,
                        &istarget);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnMultilabelMarginLoss.
  ret = aclnnMultilabelMarginLossGetWorkspaceSize(self, target, reduction, out, istarget, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultilabelMarginLossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMultilabelMarginLoss.
  ret = aclnnMultilabelMarginLoss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultilabelMarginLoss failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result in the device memory to the host side. Modify based on the specific API definition.
  auto outSize = GetShapeSize(outShape);
  std::vector<float> outData(outSize, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    outSize * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < outSize; i++) {
    LOG_PRINT("out[%ld] is: %f\n", i, outData[i]);
  }

  auto istargetSize = GetShapeSize(istargetShape);
  std::vector<float> istargetData(istargetSize, 0);
  ret = aclrtMemcpy(istargetData.data(), istargetData.size() * sizeof(istargetData[0]), istargetDeviceAddr,
                    istargetSize * sizeof(istargetData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < istargetSize; i++) {
    LOG_PRINT("istarget[%ld] is: %f\n", i, istargetData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Modify based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(out);
  aclDestroyTensor(istarget);

  // 7. Release the device resources. Modify based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(istargetDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
