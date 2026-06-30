# aclnnNanMedianDim

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/gather_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    √     |

## Function

  - Description: Returns the median and position of values in the specified dimension of a tensor after NAN is ignored.

  - Example:
    - Example 1:
      ```
      If keepDim is set to True, the size of the corresponding dimension is set to 1. If keepDim is set to False, the corresponding dimension is deleted.
      If the shape of self is [2, 3, 4], dim = 1, and keepDim is true, then the output shape is [2, 1, 4].
      If the shape of self is [2, 3, 4], dim = 1, and keepDim is false, then the output shape is [2, 4].
      ```
    - Example 2:
      ```
      Example of the output shape.
      If input
      self = tensor([[1, float('nan'), 3, 2],[-1, float('nan'), 3, 2]]) with shape [2, 4],
      dim = 0,
      keepDim = true,
      then output
      valuesOut = tensor([[-1., float('nan'), 3., 2.]]) with shape [1, 4],
      indicesOut = tensor([[1, 0, 0, 0]]) with shape [1, 4].
      ```
    - Example 3:
      ```
      If input
      self = tensor([[1, float('nan'), 3, 2],[-1, float('nan'), 3, 2]]) with shape [2, 4],
      dim = 0,
      keepDim = false,
      then output
      valuesOut = tensor([-1., float('nan'), 3., 2.]) with shape [4],
      indicesOut = tensor([1, 0, 0, 0]) with shape [4].
      ```
    - Example 4:
      ```
      If input
      self = tensor([[1, float('nan'), 3, 2],[-1, float('nan'), 3, 2]]) with shape [2, 4],
      dim = 1,
      keepDim = false,
      then output
      valuesOut = tensor ([2, 2]) with shape [2],
      indicesOut = tensor ([3, 3]) with shape [2].
      ```
    
## Prototype

  Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnNanMedianDimGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnNanMedianDim** is called to perform computation.

  - `aclnnStatus aclnnNanMedianDimGetWorkspaceSize(const aclTensor* self, int64_t dim, bool keepDim, aclTensor* valuesOut, aclTensor* indicesOut, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnNanMedianDim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnNanMedianDimGetWorkspaceSize

  - **Parameters:**

    - **self** (aclTensor*, compute input): aclTensor on the device. The shape supports zero to eight dimensions. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
      - <term>Atlas training series products</term>: The data type can be FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, or INT64.
      - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64, or BFLOAT16.
    - **dim** (int64_t, compute input): specified dimension, integer constant on the host. The value range is [–self.dim(), self.dim() – 1].
      - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: self.shape[dim] cannot be 1 when the data type of **self** is BFLOAT16.
    - **keepDim** (bool, compute input): whether to retain the dimension of the input tensor in the output tensor, bool constant on the host. If the value is **true**, the size of the corresponding dimension is set to 1. If the value is **false**, the corresponding dimension is deleted.
    - **valuesOut** (aclTensor*, compute output): median value, aclTensor on the device. The data type must be the same as that of **self**. The shape supports zero to eight dimensions. If **keepDim** is set to **true**, the shape must have the same size as that of **self** except **dim**, and the **dim** size is 1. If **keepDim** is set to **false**, the shape must be the same as that of **self** except **dim**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
      - <term>Atlas training series products</term>: The data type can be FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, or INT64.
      - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64, or BFLOAT16.
    - **indicesOut** (aclTensor*, compute output): median index, aclTensor on the device. The data type can be INT64. The shape supports zero to eight dimensions, and must be the same as that of **valuesOut**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.
    - **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.


  - **Returns:**

    **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

    ```
    The first-phase API implements input parameter verification. The following errors may be thrown:
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, valuesOut, or indicesOut is a null pointer.
    161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, valuesOut, or indicesOut is not supported.
                                      2. The data types of self and valuesOut are inconsistent.
                                      3. The dim value is out of the range [–self.dim(), self.dim() – 1].
                                      4. The number of dimensions of self, valuesOut, or indicesOut exceeds eight.
                                      5. The size of the dim dimension corresponding to self cannot be zero.
                                      6. When keepDim is set to true, the dimension of valuesOut or indicesOut is different from that of self.
                                      7. When keepDim is set to false, the dimension of valuesOut or indicesOut is not one less than that of self.
                                      8. When keepDim is set to true, the shape of valuesOut or indicesOut has a different size from that of self except dim.
                                      9. When keepDim is set to true, the shape of valuesOut or indicesOut in dim is not of size one.
                                      10. When keepDim is set to false, the shape of valuesOut or indicesOut is different from that of self except dim.
    ```

## aclnnNanMedianDim

  - **Parameters:**

    - **workspace** (void*, input): address of the workspace to be allocated on the device.
    - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnNanMedianDimGetWorkspaceSize**.
    - **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.
    - **stream** (aclrtStream, input): stream for executing the task.


  - **Returns:**

    **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).


## Constraints

- Deterministic compute:
  - **aclnnNanMedianDim** defaults to a deterministic implementation.

-  If the data type of **self** is not FLOAT, FLOAT16, or BFLOAT16, the operator execution may time out due to an overlarge tensor size (an AI CPU error is reported, with reason=[aicpu timeout]). The maximum size for each data type (closely related to the remaining memory of the machine) is as follows:
  - INT64: 150000000
  - UINT8, INT8, INT16, INT32: 725000000

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_median.h"

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
  // 2. Construct the input and output based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> valuesOutShape = {2};
  std::vector<int64_t> indicesOutShape = {2};
  void* selfDeviceAddr = nullptr;
  void* valuesOutDeviceAddr = nullptr;
  void* indicesOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* valuesOut = nullptr;
  aclTensor* indicesOut = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, NAN};
  std::vector<float> valuesOutHostData = {0, 0};
  std::vector<int64_t> indicesOutHostData = {0, 0};
  int64_t dim = 0;
  bool keepDim = false;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a valuesOut aclTensor.
  ret = CreateAclTensor(valuesOutHostData, valuesOutShape, &valuesOutDeviceAddr, aclDataType::ACL_FLOAT, &valuesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an indicesOut aclTensor.
  ret = CreateAclTensor(indicesOutHostData, indicesOutShape, &indicesOutDeviceAddr, aclDataType::ACL_INT64, &indicesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnNanMedianDim.
  ret = aclnnNanMedianDimGetWorkspaceSize(self, dim, keepDim, valuesOut, indicesOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNanMedianDimGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnNanMedianDim.
  ret = aclnnNanMedianDim(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNanMedianDim failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(valuesOutShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), valuesOutDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("valuesOut[%ld] is: %f\n", i, resultData[i]);
  }

  std::vector<int64_t> indicesData(size, 0);
  ret = aclrtMemcpy(indicesData.data(), indicesData.size() * sizeof(indicesData[0]), indicesOutDeviceAddr,
                    size * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("indicesOut[%ld] is: %ld\n", i, indicesData[i]);
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
