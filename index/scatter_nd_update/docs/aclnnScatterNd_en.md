# aclnnScatterNd
## Supported Products
[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/scatter_nd_update)


| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    √     |
| <term>Atlas training series products</term>                             |    √     |

## Function
Description: Copies the data from **data** to **out**, and updates the data in **out** based on **updates** at the specified **indices**.

## Prototype
Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnScatterNdGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnScatterNd** is called to perform computation.

* `aclnnStatus aclnnScatterNdGetWorkspaceSize(const aclTensor *data,const aclTensor *indices,const aclTensor *updates, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnScatterNd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnScatterNdGetWorkspaceSize
- **Parameters:**
  * **data** (aclTensor*, compute input): aclTensor on the device. The data type is the same as that of **updates** and **out**. The shape meets the requirement: 1 <= rank(data) <= 8. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT16, FLOAT, BOOL, or BFLOAT16.
    - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: The data type can be FLOAT16, FLOAT, or BOOL.

  * **indices** (aclTensor*, compute input): aclTensor on the device. The data type can be INT32 or INT64. The following requirements must be met: indices.shape[-1] ≤ rank(data), 1 ≤ rank(indices) ≤ 8. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. Only non-negative indices are supported. The data in **indices** cannot be out-of-bounds.

  * **updates** (aclTensor*, compute input): aclTensor on the device. The data type is the same as that of **data** and **out**. The shape must meet the following requirements: rank(updates) = rank(data) + rank(indices) – indices.shape[-1] – 1, 1 ≤ rank(updates) ≤ 8. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT16, FLOAT, BOOL, or BFLOAT16.
    - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: The data type can be FLOAT16, FLOAT, or BOOL.

  * **out** (aclTensor*, compute output): aclTensor on the device. The data type is the same as that of **data** and **out**. The shape is the same as that of **data**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT16, FLOAT, BOOL, or BFLOAT16.
    - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: The data type can be FLOAT16, FLOAT, or BOOL.

  * **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.

  * **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  ```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed data, indices, updates, or out contains a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type is not supported.
                                    2. The shape does not meet the following requirements: 1<=rank(data)<=8, 1<=rank(indices)<=8,rank(updates)=rank(data)+rank(indices)- indices.shape[-1] -1.
                                    3. The shape does not meet the following requirements: 1 ≤ rank(indices) ≤ 8, indices.shape[-1] ≤ rank(data).
                                    4. The shape does not meet the following requirements: 1 ≤ rank(updates) ≤ 8, updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1] :].
                                    5. The shape does not meet the following requirement: data.shape == out.shape.
  ```
## aclnnScatterNd
- **Parameters:**
  * **workspace** (void *, input): address of the workspace to be allocated on the device.
  * **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnScatterNdGetWorkspaceSize**.
  * **executor** (aclOpExecutor *, input): operator executor, containing the operator computation process.
  * **stream** (aclrtStream, input): stream for executing the task.
- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
  - **aclnnScatterNd** defaults to a deterministic implementation.

## Example
The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter_nd.h"
#include "aclnn/aclnn_base.h"



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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external ACL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output based on the API.
  std::vector<int64_t> dataShape = {8};
  std::vector<int64_t> indicesShape = {4, 1};
  std::vector<int64_t> updatesShape = {4};
  std::vector<int64_t> outShape = {8};

  void* dataDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* updatesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* data = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* updates = nullptr;
  aclTensor* out = nullptr;


  std::vector<float> selfHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  std::vector<int32_t> indicesData = {4,3,1,7};
  std::vector<float> updatesData = {9.0, 10.0, 11.0, 12.0};
  std::vector<float> outData = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  ret = CreateAclTensor(selfHostData, dataShape, &dataDeviceAddr, aclDataType::ACL_FLOAT, &data);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(indicesData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(updatesData, updatesShape, &updatesDeviceAddr, aclDataType::ACL_FLOAT, &updates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(outData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an out aclTensor.
  // ret = CreateAclTensor(outData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
  // CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAdd.
  ret = aclnnScatterNdGetWorkspaceSize(data, indices, updates, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterNdGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }

  ret = aclnnScatterNd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterNd failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(data);
  aclDestroyTensor(indices);
  aclDestroyTensor(updates);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.

  aclrtFree(dataDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(updatesDeviceAddr);
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
