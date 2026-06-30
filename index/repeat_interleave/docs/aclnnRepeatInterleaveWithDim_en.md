# aclnnRepeatInterleaveWithDim

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/repeat_interleave)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    √     |
| <term>Atlas training series products</term>                             |    √     |

## Function
  - Description: Repeats each element in the tensor based on the **dim** dimension for the corresponding number of times specified by the tensor **repeats**.

  - Example: Assume that the input tensor is [[a, b], [c, d], [e, f]]. **repeats** is ([1, 2, 3]), and **dim** is 0.
    In this case, the generated tensor is ([[a, b], [c, d], [c, d], [e, f], [e, f], [e, f]]).
    In the dimension with dim = 0, a and b are repeated once, c and d are repeated twice, and e and f are repeated three times.

    Assume that the input tensor is ([[a, b], [c, d], [e, f]]). **repeats** is ([2]), and **dim** is 0.
    In this case, the generated tensor is [ [a, b], [a, b], [c, d], [c, d], [e, f], [e, f]].
    In the dimension with dim = 0, a and b are repeated twice, c and d are repeated twice, and e and f are repeated twice.
    Note: This scenario is equivalent to **repeats** being (2).

## Prototype

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnRepeatInterleaveWithDimGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnRepeatInterleaveWithDim** is called to perform computation.

  - `aclnnStatus aclnnRepeatInterleaveWithDimGetWorkspaceSize(const aclTensor* self, const aclTensor* repeats, int64_t dim, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnRepeatInterleaveWithDim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnRepeatInterleaveWithDimGetWorkspaceSize

- **Parameters:**

  - **self** (aclTensor*, compute input): aclTensor on the device, indicating the input tensor to be copied. Empty tensors and [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND, and the shape supports one to eight dimensions.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, or FLOAT.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term>, <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, BFLOAT16, or FLOAT.

  - **repeats** (aclTensor*, compute input): number of repeats. aclTensor on the device. The data type can be INT32 or INT64. **repeats** can only be a zero- or one-dimensional tensor. For a one-dimensional tensor, the size of **repeats** must be 1 or equal to the size of the **dim** dimension of **self**. Empty tensors and [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.

  - **dim** (int64_t, compute input): dimension for repeats, int64_t type on the host. The value range is [–self.dim(), self.dim() – 1].

  - **outputSize** (int64_t, compute input): final size of the tensor along the **dim** dimension after repeats. int64_t type on the host. If **repeats** contains multiple values, the value of **outputSize** must be equal to the sum of the values of **repeats**. If **repeats** contains only one element, the value of **outputSize** must be equal to **repeats** multiplied by the size of the **dim** dimension of **self**.

  - **out** (aclTensor*, compute output): aclTensor on the device, output tensor after data copy is complete. The data type must be the same as that of **self**. The [data format](../../../docs/en/context/data_formats.md) can be ND. If **repeats** contains multiple values, the **dim** size of the **out** shape is equal to the sum of all elements in **repeats**. If **repeats** contains only one element, the **dim** size of the **out** shape is equal to **repeats** multiplied by the size of the **dim** dimension of **self**.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, or FLOAT.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term>, <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, BFLOAT16, or FLOAT.

  - **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.

  - **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, repeats, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self or repeats is not supported.
                                    2. The data types of self and out are inconsistent.
                                    3. repeats is not a zero- or one-dimensional tensor.
                                    4. The value of dim is not in the range of [–(number of dimensions of self), (number of dimensions of self) – 1].
                                    5. When repeats is a one-dimensional tensor, the size of repeats is neither 1 nor equal to the size of the dim dimension of self.
                                    6. The number of dimensions of self is greater than eight.
                                    7. When self is zero-dimensional, dim cannot be passed.
  ```

## aclnnRepeatInterleaveWithDim

- **Parameters:**

  - **workspace** (void*, input): address of the workspace to be allocated on the device.

  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnRepeatInterleaveWithDimGetWorkspaceSize**.

  - **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - **stream** (aclrtStream, input): stream for executing the task.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
  - **aclnnRepeatInterleaveWithDim** defaults to a deterministic implementation.

The following conditions must be met during computation:
  - **repeats** can only be a zero- or one-dimensional tensor.
    For a one-dimensional tensor, the size of **repeats** must be 1 or equal to the size of the **dim** dimension of **self**.
    The value in the **repeats** tensor must be a natural number.

  - The value of **outputSize** must meet the following conditions:
    If **repeats** contains only one element, outputSize = Size of the dim dimension of self × Value of repeats.
    If **repeats** contains multiple values, outputSize = Sum of repeats.

## Example
The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_repeat_interleave.h"

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
  std::vector<int64_t> repeatsShape = {2};
  std::vector<int64_t> outShape = {3, 3};
  void* selfDeviceAddr = nullptr;
  void* repeatsDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* repeats = nullptr;
  aclTensor* out = nullptr;
  int64_t dim = 0;
  int64_t output_size = 3;
  std::vector<int64_t> selfHostData = {3, 4, 5, -3, -4, -5};
  std::vector<int64_t> repeatsHostData = {1, 2};
  std::vector<int64_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT64, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a repeats aclTensor.
  ret = CreateAclTensor(repeatsHostData, repeatsShape, &repeatsDeviceAddr, aclDataType::ACL_INT64, &repeats);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRepeatInterleaveWithDim.
  ret = aclnnRepeatInterleaveWithDimGetWorkspaceSize(self, repeats, dim, output_size, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveWithDimGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRepeatInterleaveWithDim.
  ret = aclnnRepeatInterleaveWithDim(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveWithDim failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(repeats);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(repeatsDeviceAddr);
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
