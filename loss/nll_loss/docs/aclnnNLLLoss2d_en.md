# aclnnNLLLoss2d

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/nll_loss)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    √     |
| <term>Atlas training series products</term>                             |    √     |

## Function

- Description: Computes the negative log-likelihood loss.

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

- `aclnnStatus aclnnNLLLoss2dGetWorkspaceSize(const aclTensor *self, const aclTensor *target, const aclTensor *weight, int64_t reduction, int64_t ignoreIndex, aclTensor *out, aclTensor *totalWeightOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnNLLLoss2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnNLLLoss2dGetWorkspaceSize

- **Parameters:**

  - **self** (aclTensor*, compute input): aclTensor on the device, tensor to be computed, input x in the formula. The shape is four-dimensional, and the second dimension is C, indicating the number of classes. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.

  - **target** (aclTensor*, compute input): aclTensor on the device, real label, y in the formula. The shape has three dimensions. The first dimension of **target** is equal to the first dimension of **self**, the second dimension of **target** is equal to the third dimension of **self**, and the third dimension of **target** is equal to the fourth dimension of **self**. The value range of each element is [0, C – 1]. The data type can be INT64, UINT8, or INT32. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.

  - **weight** (aclTensor*, compute input): aclTensor on the device, scaling weight of each class, w in the formula. The shape is (C,). The data type must be the same as that of **self**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.

  - **reduction** (int64_t, compute input): int64_t on the host, reduction to be applied to the output, reduction in the formula. The value can be 0 ('none'), 1 ('mean'), or 2 ('sum'). **none** indicates that no reduction is applied; **mean** indicates that the sum of the output will be divided by the number of elements in the output; **sum** indicates that the output will be summed.

  - **ignoreIndex** (int64_t, compute input): int64_t on the host, target value that is ignored and does not affect the input gradient, ignoreIndex in the formula.

  - **out** (aclTensor*, compute output): aclTensor on the device. The data type must be the same as that of **self**. When **reduction** is **0** (**none**), the shape is the same as that of **target**. Otherwise, the shape is (1,). The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.

  - **totalWeightOut** (aclTensor*, compute output): aclTensor on the device, which indicates the sum of weights. The data type must be the same as that of **self**. The output value is valid when **reduction** is not **0** (**none**). The shape is (1,). The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.

  - **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.

  - **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, target, weight, out, or totalWeightOut is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of self, target, weight, out, or totalWeightOut is not supported.
                                    2. The data types of self, weight, out, and totalWeightOut are inconsistent.
                                    3. The shape and format of self, target, weight, out, or totalWeightOut are incorrect.
                                    4. The reduction value is not within the range of 0 to 2.
  ```

## aclnnNLLLoss2d

- **Parameters:**

  - **workspace** (void*, input): address of the workspace to be allocated on the device.

  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnNLLLoss2dGetWorkspaceSize**.

  - **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - **stream** (aclrtStream, input): stream for executing the task.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
  - **aclnnNLLLoss2d** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
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
