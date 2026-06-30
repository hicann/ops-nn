# aclnnKthvalue

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/gather_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    √    |
| <term>Atlas training series products</term>                             |    √    |

## Function

Description: Returns the kth minimum value and index of the input tensor in the specified dimension.

## Prototype
Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnKthvalueGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnKthvalue** is called to perform computation.

* `aclnnStatus aclnnKthvalueGetWorkspaceSize(const aclTensor *self, int64_t k, int64_t dim, bool keepdim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnKthvalue(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnKthvalueGetWorkspaceSize

- **Parameters:**

  - **self** (aclTensor\*, compute input): aclTensor on the device. The shape supports 1 to 8 dimensions. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported, and the [data format](../../../docs/en/context/data_formats.md) supports ND.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, or UINT8.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be BFLOAT16, FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, or UINT8.
  - **k** (int64_t, compute input): integer on the host. It indicates the kth minimum value in a specified dimension. The value range is [0, self.size(dim)].
  - **dim** (int64_t, compute input): integer on the host. It indicates the specified dimension of the input tensor. The value range is [-self.dim(), self.dim()).
  - **keepdim** (bool, compute input): bool type, indicating whether **dim** is retained in the output tensor. **True** indicates that the sizes of **valuesOut** and **indicesOut** tensors are the same as those of **self**. **False** indicates that dimension will be compressed and the obtained tensor dimension is 1 less than that of input.
  - **valuesOut** (aclTensor\*, compute output): aclTensor on the device. The data type is the same as that of **self**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported, and the [data format](../../../docs/en/context/data_formats.md) supports ND. The shape sorting axis is 1, and the non-sorting axis is the same as that of **self**.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, or UINT8.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be BFLOAT16, FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, or UINT8.
  - **indicesOut** (aclTensor\*, compute output): aclTensor on the device. The data type can be INT64. It indicates the index of the kth minimum value along the dimension **dim** in the original input tensor. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported, and the [data format](../../../docs/en/context/data_formats.md) supports ND. The shape sorting axis is 1, and the non-sorting axis is the same as that of **self**.
  - **workspaceSize** (uint64_t\*, output): size of the workspace to be allocated on the device.
  - **executor** (aclOpExecutor\**, output): operator executor, containing the operator computation process.

- **Returns:**

	**aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): The passed self, valuesOut, or indicesOut is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, valuesOut, or indicesOut is not supported, or their shapes do not match.
                                     2. The value of dim is not within the dimension range of the input tensor self.
                                     3. k is less than 0 or greater than the size of the input self in the dimension specified by dim.
  ```

## aclnnKthvalue

- **Parameters:**

  - **workspace** (void*, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnKthvalueGetWorkspaceSize**.
  - **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**

 	**aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
  - **aclnnKthvalue** defaults to a deterministic implementation.


## Example
The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_kthvalue.h"

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
  // Use CHECK as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output based on the API.
  std::vector<int64_t> selfShape = {2, 4};
  std::vector<int64_t> outShape = {2, 1};
  void* selfDeviceAddr = nullptr;
  void* valuesOutDeviceAddr = nullptr;
  void* indicesOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  uint64_t dim = 1;
  uint64_t k = 2;
  bool keepdim = true;
  aclTensor* valuesOut = nullptr;
  aclTensor* indicesOut = nullptr;
  std::vector<float> selfHostData = {0.0, 1.1, 2, 3, 4, 5, 6, 7};
  std::vector<float> valuesOutHostData = {0.0, 0};
  std::vector<int64_t> indicesOutHostData = {0, 0};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a valuesOut aclTensor.
  ret = CreateAclTensor(valuesOutHostData, outShape, &valuesOutDeviceAddr, aclDataType::ACL_FLOAT, &valuesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an indicesOut aclTensor.
  ret = CreateAclTensor(indicesOutHostData, outShape, &indicesOutDeviceAddr, aclDataType::ACL_INT64, &indicesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Modify the API as required.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnKthvalue.
  ret = aclnnKthvalueGetWorkspaceSize(self, k, dim, keepdim, valuesOut, indicesOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKthvalueGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnKthvalue.
  ret = aclnnKthvalue(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKthvalue failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> valuesData(size, 0);
  ret = aclrtMemcpy(valuesData.data(), valuesData.size() * sizeof(valuesData[0]), valuesOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("valuesResult[%ld] is: %f\n", i, valuesData[i]);
  }
  std::vector<float> indicesData(size, 0);
  ret = aclrtMemcpy(indicesData.data(), indicesData.size() * sizeof(indicesData[0]), indicesOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("indicesResult[%ld] is: %f\n", i, indicesData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(valuesOut);
  aclDestroyTensor(indicesOut);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(valuesOutDeviceAddr);
  aclrtFree(indicesOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
