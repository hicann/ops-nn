# aclnnMseLossBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/mse_loss_grad_v2)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     √   |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √   |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √   |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×   |
|  <term>Atlas inference products</term>    |     √   |
|  <term>Atlas training products</term>   |     x    |

## Function

- Description: Performs backpropagation of [aclnnMseLoss](../../mse_loss/docs/aclnnMseLoss_en.md).

- Formula:

  If `reduction` is `mean`:

  $$
  MselossBackward(grad, x, y) = grad * (x - y) * 2 / x.numel()
  $$

  `x.numel()` indicates the number of elements in `x`. If `reduction` is not `mean`:

  $$
  MselossBackward(grad, x, y) = grad * (x - y) * 2
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnMseLossBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnMseLossBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnMseLossBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, 
    const aclTensor* self,
    const aclTensor* target, 
    int64_t          reduction,
    aclTensor*       out,
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)

```Cpp
aclnnStatus aclnnMseLossBackward(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnMseLossBackwardGetWorkspaceSize

- **Parameters:**

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
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">gradOutput(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Reverse gradient input, which is the input `grad` in the formula.</td>
      <td class="tg-0pky">The shape must be broadcastable with that of self and target (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>).<br></td>
      <td class="tg-0pky">Same as self.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">self(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Input tensor, which is the input `x` in the formula.</td>
      <td class="tg-0pky">The shape must be broadcastable with that of gradOutput and target (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>).<br></td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">target(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Actual label, which is the input `y` in the formula.</td>
      <td class="tg-0pky">The shape must be broadcastable with that of gradOutput and self (see <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">Broadcast Relationship</a>).<br></td>
      <td class="tg-0pky">The value is the same as that of self.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">reduction (int64_t) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Specifies the reduction to be applied to the output. The parameter `reduction` in the formula is used.</td>
      <td class="tg-0pky">0 ('none'), 1 ('mean'), or 2 ('sum') is supported.<br>- none: No reduction is applied.<br>- mean: The sum of the output is divided by the number of elements in the output.<br>- sum: The output is summed. </td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">out(aclTensor*)</td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Output loss, which is the output `MselossBackward(grad, x, y)` in the formula.</td>
      <td class="tg-0pky">The shape of `out` must be consistent with the shape of the tensor obtained after <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcasting</a> `gradOutput`, `self`, and `target`.<br>The data type must be convertible from BOOL (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</td>
      <td class="tg-0pky">FLOAT, FLOAT16, BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
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
      <td class="tg-0pky">Returns the operator executor, including the operator computation process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

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
      <td class="tg-0pky">The input gradOutput, self, target, or out is a null pointer.</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="6">161002</td>
      <td class="tg-0pky">The data type of self is not supported.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The data types of gradOutput and target are different from that of self.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shapes of gradOutput, self, and target are not <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcastable</a>.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shape of gradOutput, self, and target after <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcasting</a> is inconsistent with the shape of out.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The value of reduction is not in the range of 0 to 2.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The shape of gradOutput, self, or target exceeds eight dimensions.</td>
    </tr>
  </tbody>
  </table>

## aclnnMseLossBackward

- **Parameters:**

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
          <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnMseLossBackwardGetWorkspaceSize API.</td>
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
    - **aclnnMseLossBackward** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mse_loss_backward.h"

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

  // 2. Construct the input and output, and customize gradOutput based on the API.
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> targetShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> gradOutputHostData = {0, 1, 2, 3};
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> targetHostData = {1, 1, 1, 1};
  std::vector<float> outHostData(4, 0);
  // Create a gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr,
                        aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a reduction.
  int64_t reduction = 1;

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnMseLossBackward.
  ret = aclnnMseLossBackwardGetWorkspaceSize(gradOutput, self, target, reduction, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMseLossBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMseLossBackward.
  ret = aclnnMseLossBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMseLossBackward failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(gradOutputDeviceAddr);
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
