# aclnnInplaceScatterUpdate

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/scatter)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    ×    |
| <term>Atlas training series products</term>                             |    √    |

## Function

- Description:
  Updates the values in **data** one by one by referring to the values in **updates** based on the specified **axis** and **indices**. The operator semantics are customized without corresponding TensorFlow or PyTorch APIs.

- Example:
  This operator has three inputs and one attribute: **data**, **updates**, **indices**, and **axis**. **data** is the tensor to be updated, **updates** is the tensor containing the update data, and **indices** specifies the update locations.
  **axis** specifies the update dimension. When **indices** is one-dimensional, there are two scenarios:

  **Scenario 1:** When **indices** is one-dimensional, **axis** specifies that the shape of the update dimension is 1 and **indices** specifies the offset of each batch dimension (the highest dimension) in the **axis** dimension.

  ```
  Input example:
  data:(a, b, c, d)
  updates:(a, b, 1, d)
  indices:(a,)
  axis = -2
  ```

      data[i][j][indices[i]][k] = updates[i][j][0][k] # if dim=-2
      data[i][j][k][indices[i]] = updates[i][j][k][0] # if dim=-1

  **Scenario 2:** When **indices** is one-dimensional, **axis** specifies that the shape of the update dimension is greater than 1 and **indices** specifies the offset of each batch dimension (the highest dimension) in the **axis** dimension.

  ```
  Input example:
  data:(a, b, c, d)
  updates:(a, b, e, d), indices[i] + e <= c
  indices:(a,)
  axis = -2 or 2
  ```

      data[i][j][indices[i]+k][l] = updates[i][j][k][l] # if dim=-2
      data[i][j][k][indices[i]+l] = updates[i][j][k][l] # if dim=-1

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnInplaceScatterUpdateGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnInplaceScatterUpdate** is called to perform computation.

* `aclnnStatus aclnnInplaceScatterUpdateGetWorkspaceSize(aclTensor *data, const aclTensor *indices, const aclTensor *updates, int64_t axis, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnInplaceScatterUpdate(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnInplaceScatterUpdateGetWorkspaceSize

- **Parameters:**

  * **data** (aclTensor*, compute input | compute output): It supports 2 to 8 dimensions, and the number of dimensions must be the same as that of **updates**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. Empty tensors are not supported.
    * <term>Atlas training series products</term>: The data type can be INT8, FLOAT16, FLOAT32, or INT32.
    * <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be INT8, FLOAT16, FLOAT32, INT32, or BFLOAT16.
  * **indices** (aclTensor*, compute input): The data type can be INT32 or INT64. Currently, only zero, one, and two dimensions are supported. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. Empty tensors are not supported. Only non-negative indices are supported. The data in **indices** cannot be out-of-bounds.
  * **updates** (aclTensor*, compute input): The data type must be the same as that of **data**, and the number of dimensions in the shape must be the same as that of the **data** shape. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. Empty tensors are not supported.
    * <term>Atlas training series products</term>: The data type can be INT8, FLOAT16, FLOAT32, or INT32.
    * <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be INT8, FLOAT16, FLOAT32, INT32, or BFLOAT16.
  * **axis** (int64_t, compute input): dimension to scatter. The data type is INT64. The value range is (-data_rank, data_rank) (data_rank indicates the number of dimensions of **data**). **axis** cannot be 0.
  * **workspaceSize** (uint64_t *, output): size of the workspace to be allocated on the device.
  * **executor** (aclOpExecutor **, output): operator executor, containing the operator computation process.

- **Returns:**
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed data, indices, or updates is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of data, indices, or updates is not supported.
                                    2. The data types of data and updates are inconsistent.
                                    3. The number of dimensions of data is inconsistent with that of updates.
                                    4. The dimension of indices is not zero, one, or two.
                                    5. When the dimension of indices is 0, the 0th axis of updates is not 1.
                                    6. data, indices, and updates are empty tensors.
  ```

## aclnnInplaceScatterUpdate

- **Parameters:**
  * **workspace** (void*, input): address of the workspace to be allocated on the device.
  * **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnInplaceScatterUpdateGetWorkspaceSize**.
  * **executor** (aclOpExecutor *, input): operator executor, containing the operator computation process.
  * **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
  - **aclnnInplaceScatterUpdate** defaults to a deterministic implementation.

- The 0th axis of the updates shape must be consistent with that of the indices shape.
- If indices is zero-dimensional, the 0th axis of the updates shape must be 1.
- The 0th axis of the updates shape must be less than or equal to that of the data shape.
- The shapes of updates and data are the same except for the axis and 0th axis.
- When the indices shape is two-dimensional, the 1st axis of the shape must be 2.
- If the data type of indices is INT32, DtypeSize is 4. If the data type of indices is INT64, DtypeSize is 8. IndicesShapeSize is the product of the indices shape. The required UB is calculated as follows: UB = IndicesShapeSize x DtypeSize + 224. If the required UB size is greater than the total UB size of the corresponding AI processor version, the operation is not supported.
- If indices contains duplicates, the output at those positions is non-deterministic.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter_update.h"

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
  int64_t axis = -2;
  std::vector<int64_t> selfRefShape = {1, 1, 2, 8};
  std::vector<int64_t> indicesShape = {1};
  std::vector<int64_t> updatesShape = {1, 1, 1, 8};
  void* selfRefDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* updatesDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* updates = nullptr;
  std::vector<float> selfRefHostData = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int64_t> indicesHostData = {1};
  std::vector<float> updatesHostData = {3, 3, 3, 3, 3, 3, 3, 3};

  // Create a selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an updates aclTensor.
  ret = CreateAclTensor(updatesHostData, updatesShape, &updatesDeviceAddr, aclDataType::ACL_FLOAT, &updates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceScatterUpdate.
  ret = aclnnInplaceScatterUpdateGetWorkspaceSize(selfRef, indices, updates, axis, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceScatterUpdate.
  ret = aclnnInplaceScatterUpdate(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterUpdate failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(selfRef);
  aclDestroyTensor(indices);
  aclDestroyTensor(updates);

  // 7. Release device resources.
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(updatesDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
