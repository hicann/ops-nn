# aclnnIndexPutImpl

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/index_put_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    √     |

## Function

- Description: Replaces or accumulates the coordinate data of input **self** and the input **values** based on **indices**.
- Formula:

  - If **accumulate** is set to **False**:

    $$
    self[indices] = values
    $$

  - If **accumulate** is set to **True**:
    
    $$
    self[indices]  = self[indices]  + values
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnIndexPutImplGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnIndexPutImpl** is called to perform computation.

- `aclnnStatus aclnnIndexPutImplGetWorkspaceSize(aclTensor* selfRef, const aclTensorList* indices, const aclTensor* values, const bool accumulate, const bool unsafe, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnIndexPutImpl(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnIndexPutImplGetWorkspaceSize

* **Parameters:**
  
  * **selfRef** (aclTensor*, computation input/output): $self$ in the formula and aclTensor on the device. The data type is the same as that of **values**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. The shape can be 1D to 8D.
    - <term>Atlas training series products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, or BOOL.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, or BFLOAT16.

  * **indices** (aclTensorList*, computation input): $indices$ in the formula and aclTensorList on the device. The data type can be INT32, INT64, or BOOL. The [data format](../../../docs/en/context/data_formats.md) can be ND.

  * **values** (aclTensor*, computation input): $values$ in the formula and aclTensor on the device. The data type is the same as that of **selfRef**. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas training series products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, or BOOL.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, or BFLOAT16.

  * **accumulate** (bool, computation input): accumulation or update operation type. **True** indicates accumulation, and **False** indicates update. Boolean value on the host.
  * **unsafe** (bool, computation input): Boolean value on the host, checking whether the index is within the valid range. If **unsafe** is set to **True** and an index is out of bounds, an error is reported and the execution exits. If **unsafe** is set to **False** and an index is out of bounds, an exception may occur during running.

  * **workspaceSize** (uint64_t\*, output): size of the workspace to be allocated on the device.

  * **executor** (aclOpExecutor\**, output): operator executor, containing the operator computation process.

* **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  
  ```
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): The passed selfRef, indices, or values is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of selfRef or values is not supported.
                                        2. The data types of selfRef and values are inconsistent.
                                        3. The data formats of selfRef and values are inconsistent.
  ```

## aclnnIndexPutImpl

- **Parameters:**
  
  - **workspace** (void\*, input): address of the workspace to be allocated on the device.

  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling first-phase API **aclnnIndexPutImplGetWorkspaceSize**.

  - **executor** (aclOpExecutor\*, input): operator executor, containing the operator computation process.

  - **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic computation:
  - **aclnnIndexPutImpl** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic computation.

- The input parameters **selfRef**, **indices**, and **values** have the following constraints:
  - 1. The number of tensors in **indices** cannot exceed the dimensions of **selfRef**.
  - 2. The dimensions of **values** must meet the following formula or meet the following formula after broadcasting:
      - values.Dims() = indices[i].Dims() + (selfRef.Dims() - indices.size())
      - The first half of the dimensions of **values** must be the same as the tensor dimensions in **indices** (the tensors in **indices** are broadcast to the same shape), and the second half of the dimensions must be the same as the dimensions of **selfRef** minus the number of tensors in **indices**.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_index_put_impl.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {3, 4};
  std::vector<int64_t> indexShape = {1};
  std::vector<int64_t> valueShape = {1};
  void* selfDeviceAddr = nullptr;
  void* indexOneDeviceAddr = nullptr;
  void* indexTwoDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* indexOne = nullptr;
  aclTensor* indexTwo = nullptr;
  aclTensor* value= nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> valueHostData = {1};
  std::vector<int64_t> indexOneHostData = {0};
  std::vector<int64_t> indexTwoHostData = {2};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an index aclTensor.
  ret = CreateAclTensor(indexOneHostData, indexShape, &indexOneDeviceAddr, aclDataType::ACL_INT64, &indexOne);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(indexTwoHostData, indexShape, &indexTwoDeviceAddr, aclDataType::ACL_INT64, &indexTwo);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* indexs[] = {indexOne, indexTwo};
  auto indexTensorList = aclCreateTensorList(indexs, 2);
  // value aclTensor
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnIndexPutImpl API call example
  // 3. Call the CANN operator library API, which needs to be replaced with a specific operator API.
  // Call the first-phase API of aclnnIndexPutImpl.
  ret = aclnnIndexPutImplGetWorkspaceSize(self, indexTensorList, value, true, false, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexPutImplGetWorkspaceSizefailed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnIndexPutImpl.
  ret = aclnnIndexPutImpl(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexPutImplfailed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensorList(indexTensorList);
  aclDestroyTensor(value);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(indexOneDeviceAddr);
  aclrtFree(indexTwoDeviceAddr);
  aclrtFree(valueDeviceAddr);
  if (workspaceSize > 0) {
   aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
