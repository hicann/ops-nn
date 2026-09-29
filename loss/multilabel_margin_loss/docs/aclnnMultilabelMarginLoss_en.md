# aclnnMultilabelMarginLoss

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/multilabel_margin_loss)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     x    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √   |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √   |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×   |
|  <term>Atlas inference products</term>    |     ×   |
|  <term>Atlas training products</term>   |     √   |

## Function

- Function: Calculates the negative log likelihood loss.
- Formula:
**self** is the input, with shape (N, C) or (C), where **N** indicates the batch size and **C** indicates the number of classes. **target** indicates the real label, with shape (N, C) or (C). The value range of each element is [-1, C - 1]. To ensure that the shape is the same as that of the input, -1 is padded. That is, the label before the first -1 indicates the real label **yTrue** to which the sample belongs. For example, if y = [0,3,-1,1], the real label yTrue is [0,3]. The formula for calculating each sample is as follows:

  $$
    istarget[k]=\begin{cases}
      \ 1, &
      \text{k in yTrue}\\
      \ 0, &
      \text{otherwise}
  \end{cases}
  $$

  $$
    l_n=\sum^C_{j,istarget[j]=1}\sum^C_{i,istarget[i]=0} \frac{max(0,1-x[j]-x[i])}{C}
  $$

  If `reduction` is `none`:

  $$
  \ell(x, y) = L = \{l_1,\dots,l_N\}^\top
  $$

  If `reduction` is not `none`:

  $$
  \ell(x, y) = \begin{cases}
      \sum_{n=1}^N \frac{1}{N} l_n, &
      \text{if reduction} = \text{mean;}\\
      \sum_{n=1}^N l_n, &
      \text{if reduction} = \text{sum.}
  \end{cases}
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnMultilabelMarginLossGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnMultilabelMarginLoss** is called to perform computation.

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
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-consecutive Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor, which is the input x in the formula.</td>
      <td class="tg-0pky">The shape is (N, C) or (C), where N indicates the batch size and C indicates the number of classes.<br>If the number of elements in `self` is greater than 15,000 x 20,000, error 507034 may be reported, indicating that the Vector Core execution times out.</td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1, 2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">target (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Actual label, which is the input y in the formula.</td>
      <td class="tg-0pky">The shape is (N, C) or (C). The value range of each element is [-1, C – 1]. The value -1 is used for padding. That is, the label before the first -1 indicates the actual label of the sample.</td>
      <td class="tg-0pky">INT32, INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1, 2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">reduction (int64_t) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Reduction to be applied to the output, which is the reduction parameter in the formula.</td>
      <td class="tg-0pky">The value can be 0 (none), 1 (mean), or 2 (sum).<br>'none' indicates that no reduction is applied.<br>'mean' indicates that the sum of the output will be divided by the number of elements in the output.<br>'sum' indicates that the output will be summed up.</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">out (aclTensor*) </td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Output loss, which is ℓ(x,y) in the formula.</td>
      <td class="tg-0lax">The shape is (N) or ()</td>.
      <td class="tg-0lax">The value is the same as that of self and isTarget.</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">0, 1</td>
      <td class="tg-0lax"></td>
    </tr>
    <tr>
      <td class="tg-0pky">isTarget (aclTensor*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Output `istarget `</td> in the formula.
      <td class="tg-0pky">The shape is (N, C) or (C)</td>.
      <td class="tg-0pky">The value is the same as that of self and out.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1, 2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size to be allocated on the device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the operator executor, which contains the operator computation process.</td>
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
      <td class="tg-0pky" rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="5">161002</td>
      <td class="tg-0pky">The data types of self, out, and isTarget are not supported.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The data types of self, out, and isTarget are inconsistent.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The data type of target is not supported.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shapes of self, target, and isTarget do not meet the requirements specified in the parameter description.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shape of out does not meet the requirements specified in the parameter description.</td>
    </tr>
  </tbody>
  </table>

## aclnnMultilabelMarginLoss

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
          <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnMultilabelMarginLossGetWorkspaceSize API.</td>
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
    - **aclnnMultilabelMarginLoss** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT32, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  //Create an istarget aclTensor.
  ret = CreateAclTensor(istargetHostData, istargetShape, &istargetDeviceAddr, aclDataType::ACL_FLOAT,
                        &istarget);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnMultilabelMarginLoss.
  ret = aclnnMultilabelMarginLossGetWorkspaceSize(self, target, reduction, out, istarget, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultilabelMarginLossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMultilabelMarginLoss.
  ret = aclnnMultilabelMarginLoss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultilabelMarginLoss failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
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

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(out);
  aclDestroyTensor(istarget);

  // 7. Release device resources. Modify the configuration based on the API definition.
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
