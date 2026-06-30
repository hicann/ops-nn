# aclnnUniqueConsecutive

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/unique_consecutive)

## Supported Products
| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √       |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |   √     |
| <term>Atlas training series products</term>                             |   √     |
## Function

Description: Eliminates all but the first element from every consecutive group of equivalent elements. When **dim** is not empty, tensors except the first tensor from every consecutive group of equivalent elements are eliminated along the corresponding dimension.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnUniqueConsecutiveGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnUniqueConsecutive** is called to perform computation.

- `aclnnStatus aclnnUniqueConsecutiveGetWorkspaceSize(const aclTensor* self, bool returnInverse, bool returnCounts, int64_t dim, aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnUniqueConsecutive(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnUniqueConsecutiveGetWorkspaceSize

* **Parameters:**
  - **self** (aclTensor*, compute input): aclTensor on the device. The number of dimensions cannot exceed eight. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128 or BOOL.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term>, <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128, BOOL or BFLOAT16.
  - **returnInverse** (bool, compute input): whether to return the position subscripts of elements in **valueOut** corresponding to elements of **self**. **True** indicates that the subscripts are returned, and **False** indicates that the subscripts are not returned.
  - **returnCounts** (bool, compute input): whether to return the number of consecutive occurrences for where elements in the input **self** map to in **valueOut**. **True** indicates that the number is returned, and **False** indicates that the number is not returned.
  - **dim** (int64_t, compute input): dimension to apply unique.
  - **valueOut** (aclTensor*, compute output): first output tensor, which returns the result after consecutive duplicate elements are eliminated. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128 or BOOL.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128, BOOL or BFLOAT16.
  - **inverseOut** (aclTensor*, compute output): second output tensor. This parameter is valid when **returnInverse** is **True**. It returns the position subscripts of elements in **valueOut** corresponding to elements of **self**. The data type can be INT64. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **countsOut** (aclTensor*, compute output): third output tensor. This parameter is valid when **returnCounts** is **True**. It returns the number of consecutive occurrences for where elements in the input **self** map to in **valueOut**. The data type can be INT64. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **workspaceSize** (uint64_t\*, output): size of the workspace required to be allocated on the device.
  - **executor** (aclOpExecutor\**, output): operator executor, containing the operator computation process.
* **Returns:**
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

```
The first-phase API implements input parameter verification. The following errors may be thrown:
161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, valueOut, inverseOut, or countsOut is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self is not supported.
                                     2. The data types of self and valueOut are inconsistent.
                                     3. The data type of inverseOut or countsOut is not supported.
                                     4. The data types of inverseOut and countsOut are inconsistent.
```

## aclnnUniqueConsecutive

* **Parameters:**
  - **workspace** (void\*, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64\_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnUniqueConsecutiveGetWorkspaceSize**.
  - **executor** (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
  - **aclnnUniqueConsecutive** defaults to a deterministic implementation.


## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_unique_consecutive.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
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
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> valueShape = {8};
  std::vector<int64_t> inverseShape = {4, 2};
  std::vector<int64_t> countsShape = {8};
  void* selfDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  void* inverseDeviceAddr = nullptr;
  void* countsDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* valueOut = nullptr;
  aclTensor* inverseOut = nullptr;
  aclTensor* countsOut = nullptr;
  std::vector<float> selfHostData = {0, 1, 1, 3, 3, 1, 1, 3};
  std::vector<float> valueHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> inverseHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> countsHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  bool returnInverse = false;
  bool returnCounts = false;
  int64_t dim = 0;

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a valueOut aclTensor.
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &valueOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an inverseOut aclTensor.
  ret = CreateAclTensor(inverseHostData, inverseShape, &inverseDeviceAddr, aclDataType::ACL_INT64, &inverseOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a countsOut aclTensor.
  ret = CreateAclTensor(countsHostData, countsShape, &countsDeviceAddr, aclDataType::ACL_INT64, &countsOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnUniqueConsecutive.
  ret = aclnnUniqueConsecutiveGetWorkspaceSize(self, returnInverse, returnCounts, dim, valueOut, inverseOut, countsOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUniqueConsecutiveGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnUniqueConsecutive.
  ret = aclnnUniqueConsecutive(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUniqueConsecutive failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(valueShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), valueDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(valueOut);
  aclDestroyTensor(inverseOut);
  aclDestroyTensor(countsOut);
  return 0;

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(inverseDeviceAddr);
  aclrtFree(countsDeviceAddr);
  if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
