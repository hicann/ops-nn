# aclnnNLLLoss2d

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/nll_loss)

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

- This API is used to calculate the negative log likelihood loss.

- Formula:

  If `reduction` is `none`:

  $$
  \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
  l_n = - w_{y_n} x_{n,y_n}, \quad
  w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignoreIndex}\},
  $$

  $x$ indicates self, $y$ indicates target, $w$ indicates weight, and $N$ indicates the batch size. If `reduction` is not `none`:

  $$
  \ell(x, y) = \begin{cases}
      \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
      \text{if reduction} = \text{`mean`}\\
      \sum_{n=1}^N l_n,  &
      \text{if reduction} = \text{`sum`}
  \end{cases}
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnNLLLoss2dGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnNLLLoss2d** is called to perform computation.

```cpp
aclnnStatus aclnnNLLLoss2dGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *target,
    const aclTensor *weight,
    int64_t          reduction,
    int64_t          ignoreIndex,
    aclTensor       *out,
    aclTensor       *totalWeightOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnNLLLoss2d(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnNLLLoss2dGetWorkspaceSize

- **Parameters**

   <table style="undefined;table-layout: fixed; width: 1446px"><colgroup>
    <col style="width: 158px">
    <col style="width: 120px">
    <col style="width: 290px">
    <col style="width: 320px">
    <col style="width: 218px">
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
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>Tensor to be computed.</td>
        <td>The input x in the formula is a 4D tensor, and the second dimension is C, which indicates the number of classes.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>target (aclTensor*) </td>
        <td>Input</td>
        <td>Actual label.</td>
        <td><ul><li>The input y in the formula is a 3D tensor.</li><li>The first dimension of target is equal to the first dimension of self, the second dimension of target is equal to the third dimension of self, and the third dimension of target is equal to the fourth dimension of self.</li><li>The value range of each element is [0, C - 1].</li></ul></td>
        <td>INT64, UINT8, INT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight (aclTensor*) </td>
        <td>Input</td>
        <td>Scaling weight for each class.</td>
        <td>w in the formula, whose shape is (C,).</td>
        <td>The data type is the same as that of self.</td>
        <td>ND</td>
        <td>(C,)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduction (int64_t) </td>
        <td>Input</td>
        <td>Reduction to be applied to the output.</td>
        <td><ul>The value can be 0 (none), 1 (mean), or 2 (sum). <li>'none' indicates that no reduction is applied.</li><li>'mean' indicates that the sum of the output is divided by the number of elements in the output.</li><li>'sum' indicates that the output is summed up.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ignoreIndex (int64_t) </td>
        <td>Input</td>
         <td>A target value to be ignored and does not affect the input gradient.
        </td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>`out` in the formula.</td>
        <td>When reduction is 0 (none), the shape is the same as that of target. Otherwise, the shape is (1,).</td>
        <td>The data type is the same as that of self.</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>totalWeightOut (aclTensor*) </td>
        <td>Output</td>
        <td>totalWeightOut in the formula.</td>
        <td>The output value is valid when reduction is not 0 (not none). The shape is (1,).</td>
        <td>The data type is the same as that of self.</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
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

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
      <col style="width: 300px">
      <col style="width: 134px">
      <col style="width: 716px">
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
      <td>The input self, target, weight, out, or totalWeightOut is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type or format of self, target, weight, out, or totalWeightOut is not supported.</td>
      </tr>
      <tr>
      <td>The data types of self, weight, out, or totalWeightOut are inconsistent.</td>
      </tr>
       <tr>
      <td>The shape and format of self, target, weight, out, or totalWeightOut are incorrect.</td>
      </tr>
      <tr>
      <td>The value of reduction is not in the range of 0 to 2.</td>
      </tr>
      </tbody>
      </table>

## aclnnNLLLoss2d

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
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnNLLLoss2dGetWorkspaceSize API.</td>
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
    - **aclnnNLLLoss2d** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_nll_loss2d.h"

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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
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
  std::vector<int64_t> selfShape = {1, 2, 3, 2};
  std::vector<int64_t> targetShape = {1, 3, 2};
  std::vector<int64_t> weightShape = {2};
  std::vector<int64_t> outShape = {1, 3, 2};
  std::vector<int64_t> totalWeightOutShape = {1};
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* totalWeightOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* out = nullptr;
  aclTensor* totalWeightOut = nullptr;
  std::vector<float> selfHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1};
  std::vector<int32_t> targetHostData = {1, 0, 1, 1, 2, 1};
  std::vector<float> weightHostData = {1.1, 1.2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0};
  std::vector<float> totalWeightOutHostData = {0};
  int64_t reduction = 0;
  int64_t ignoreIndex = -100;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT32, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a totalWeightOut aclTensor.
  ret = CreateAclTensor(totalWeightOutHostData, totalWeightOutShape, &totalWeightOutDeviceAddr, aclDataType::ACL_FLOAT,
                        &totalWeightOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnNLLLoss2d.
  ret = aclnnNLLLoss2dGetWorkspaceSize(self, target, weight, reduction, ignoreIndex, out, totalWeightOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLoss2dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnNLLLoss2d.
  ret = aclnnNLLLoss2d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLoss2d failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(weight);
  aclDestroyTensor(out);
  aclDestroyTensor(totalWeightOut);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(totalWeightOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
