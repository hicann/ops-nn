# aclnnRepeatInterleaveInt

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
  - Description: Repeats each element for the number of times specified by **repeats** after the tensor **self** is flattened.

  - Example:
    Assume that the input tensor is ([[a, b], [c, d], [e, f]]), and the value of **repeats** is 2.
    In this case, the generated tensor is ([a, a, b, b, c, c, d, d, e, e, f, f]).
    After the tensor is flattened, the input is converted into ([a, b, c, d, e, f]). Each element in the tensor is repeated based on the number of times specified by **repeats**, that is, each element is repeated twice.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnRepeatInterleaveIntGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnRepeatInterleaveInt** is called to perform computation.

  - `aclnnStatus aclnnRepeatInterleaveIntGetWorkspaceSize(const aclTensor* self, int64_t repeats, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnRepeatInterleaveInt(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnRepeatInterleaveIntGetWorkspaceSize

- **Parameters:**

  - **self** (aclTensor*, compute input): aclTensor on the device, indicating the input tensor to be copied. Empty tensors and [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND, and the shape supports zero to eight dimensions.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, or FLOAT.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term>, <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, BFLOAT16, or FLOAT.

  - **repeats** (int64_t, compute input): number of repeats. int64_t type on the host. The value of **repeats** must be a non-negative number.

  - **outputSize** (int64_t, compute input): final size of the tensor after repeats. int64_t type on the host. The value of **outputSize** must be equal to the number of elements of **self** multiplied by the value of **repeats**.

  - **out** (aclTensor*, compute output): aclTensor on the device, output tensor after data copy is complete. The data type must be the same as that of **self**. The [data format](../../../docs/en/context/data_formats.md) can be ND. The shape is a one-dimensional tensor, and must have the same size as **outputSize**.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, or FLOAT.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term>, <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, BFLOAT16, or FLOAT.

  - **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.

  - **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self is not supported.
                                    2. The data types of self and out are inconsistent.
                                    3. The value of repeats is not a natural number.
                                    4. The number of dimensions of self is greater than eight.
  ```

## aclnnRepeatInterleaveInt

- **Parameters:**

  - **workspace** (void*, input): address of the workspace to be allocated on the device.

  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnRepeatInterleaveIntGetWorkspaceSize**.

  - **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - **stream** (aclrtStream, input): stream for executing the task.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
  - **aclnnRepeatInterleaveInt** defaults to a deterministic implementation.

The following condition must be met during computation: outputSize = Value of repeats × Number of elements.

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
  std::vector<int64_t> outShape = {12};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  int64_t repeats = 2;
  int64_t output_size = 12;
  std::vector<float> selfHostData = {3, 4, 5, -3, -4, -5};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRepeatInterleaveInt.
  ret = aclnnRepeatInterleaveIntGetWorkspaceSize(self, repeats, output_size, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveIntGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRepeatInterleaveInt.
  ret = aclnnRepeatInterleaveInt(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveInt failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
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
