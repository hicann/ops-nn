# aclnnL1Loss

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/lp_loss)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    √    |
| <term>Atlas training series products</term>                             |    √    |

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnL1LossGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnL1Loss** is called to perform computation.

- `aclnnStatus aclnnL1LossGetWorkspaceSize(const aclTensor* self, const aclTensor* target, int64_t reduction, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnL1Loss(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## Function

- Description: Computes the mean absolute error (MAE) between each element of input self and target. **reduction** specifies the reduction to be applied to the output. The value can be **none**, **mean**, or **sum**. **none** indicates that no reduction is applied; **mean** indicates that the sum of the output will be divided by the number of elements in the output; **sum** indicates that the output will be summed.

- Formula:

  If `reduction` is `none`:
  
  $$
  \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
  l_n = \left| x_n - y_n \right|,
  $$
  
  $x$ indicates self, $y$ indicates target, and $N$ indicates the batch size. If `reduction` is not `none`:
  
  $$
  \ell(x, y) =
  \begin{cases}
      \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
      \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
  \end{cases}
  $$

## aclnnL1LossGetWorkspaceSize

- **Parameters:**

  - **self** (aclTensor*, compute input): input `self` in the formula, aclTensor on the device. The data types of **self** and **target** must meet the type deduction rules (see [Deduction Relationship](../../../docs/en/context/deduction_relationship.md)). The shape supports 0 to 8 dimensions. The shapes of **self** and **target** must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported, and the [data format](../../../docs/en/context/data_formats.md) supports ND.
    - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: The data type can be FLOAT16, FLOAT32, or INT64.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be BFLOAT16, FLOAT16, FLOAT32, or INT64.
  - **target** (aclTensor*, compute input): input `target` in the formula, aclTensor on the device. The data types of **target** and **self** must meet the type deduction rules (see [Deduction Relationship](../../../docs/en/context/deduction_relationship.md)). The shape supports 0 to 8 dimensions. The shapes of **target** and **self** must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md). [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported, and the [data format](../../../docs/en/context/data_formats.md) supports ND.
    - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: The data type can be FLOAT16, FLOAT32, or INT64.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be BFLOAT16, FLOAT16, FLOAT32, or INT64.
  - **reduction** (int64_t, compute input, attribute): input to be applied to the reduction formula of the output, `reduction` in the formula, an integer on the host. The value can be 0 ('none') | 1 ('mean') | 2 ('sum'). **none** indicates that no reduction is applied; **mean** indicates that the sum of the output will be divided by the number of elements in the output; **sum** indicates that the output will be summed.
  - **out** (aclTensor*, compute output): `out` in the formula, aclTensor on the device. The data type must be convertible from that after deduction between **self** and **target** (see [Conversion Relationship](../../../docs/en/context/conversion_relationship.md)). [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported, and the [data format](../../../docs/en/context/data_formats.md) supports ND. When **reduction** is 0, the shape of **out** is the same as the broadcasted shape of **self** and **target**. When **reduction** is not 0, the shape of **out** supports 0 dimensions.
    - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: The data type can be FLOAT16, FLOAT32, INT64, COMPLEX64, or COMPLEX128.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be BFLOAT16, FLOAT16, FLOAT32, INT64, COMPLEX64, or COMPLEX128.
  - **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.
  - **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, target, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data types of self and target do not meet the type deduction rules, or the deduced data type is not supported.
                                        2. The deduced data types of self and target cannot be cast to the data type of out.
                                        3. self or target has more than 8 dimensions.
                                        4. The shapes of self and target do not meet the broadcast rules.
                                        5. The reduction value is not within the range of 0 to 2.
                                        6. When reduction is 0, the broadcasted shape of self and target is different from the shape of out.
                                        7. When reduction is not 0, the dimension of out is greater than 0.
                                        8. When reduction is none and self is not a floating-point number, target cannot be a floating-point number.
                                        9. When reduction is mean, at least one of self and target must be a floating-point number.
  ```

## aclnnL1Loss

- **Parameters:**

  - **workspace** (void*, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnL1LossGetWorkspaceSize**.
  - **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic compute:
  - **aclnnL1Loss** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_l1_loss.h"

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
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> targetShape = {2, 2};
  std::vector<int64_t> outShape = {};
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> targetHostData = {1, 1, 1, 1};
  std::vector<float> outHostData = {0};
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
  // Call the first-phase API of aclnnL1Loss.
  ret = aclnnL1LossGetWorkspaceSize(self, target, reduction, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnL1LossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnL1Loss.
  ret = aclnnL1Loss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnL1Loss failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
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
