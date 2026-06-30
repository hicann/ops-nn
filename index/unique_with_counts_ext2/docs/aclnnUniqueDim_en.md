# aclnnUniqueDim

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/unique_with_counts_ext2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×      |
| <term>Atlas training series products</term>                             |    √     |

## Function

- Description: Removes duplicates from the input tensor `self` along a given dim.
- Example: Assume that `self` is
  
  $$
  \begin{bmatrix}
  \begin{bmatrix}
   2 & 1 & 2 
  \end{bmatrix}\\
   \begin{bmatrix}
   1 & 2 & 1 
  \end{bmatrix}\\
   \begin{bmatrix}
   2 & 1 & 2 
  \end{bmatrix}\\
  \end{bmatrix}
  $$
  
  and `dim` is 0, then `valueOut` is:
  
  $$
  \begin{bmatrix}
  \begin{bmatrix}
   2 & 1 & 2 
  \end{bmatrix}\\
   \begin{bmatrix}
   1 & 2 & 1 
  \end{bmatrix}\\ 
  \end{bmatrix}
  $$
  
  `inverseOut` is:
  
  $$
  \begin{bmatrix}
   0 & 1 & 0 
  \end{bmatrix}
  $$
  
  `countsOut` is:
  
  $$
  \begin{bmatrix}
   1 & 2  
  \end{bmatrix}
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnUniqueDimGetWorkspaceSize** is called to obtain the input parameters and to compute the workspace size required for computation. Then, **aclnnUniqueDim** is called to perform computation.

- `aclnnStatus aclnnUniqueDimGetWorkspaceSize(const aclTensor* self, bool sorted, bool returnInverse, int64_t dim, aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnUniqueDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnUniqueDimGetWorkspaceSize

- **Parameters:**
  
  - **self** (aclTensor\*, compute input): `self` in the example, aclTensor on the device. The shape can be one- to eight-dimensional. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas training series products</term>: The data type can be FLOAT, FLOAT16, UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, DOUBLE, or BOOL.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, DOUBLE, BOOL, or BFLOAT16.
  - **sorted** (bool, compute input): whether the returned output result `valueOut` is sorted.
  - **returnInverse** (bool, compute input): whether to return the position subscripts of elements in **valueOut** corresponding to elements of **self** on the dim axis. **True** indicates that the subscripts are returned, and **False** indicates that the subscripts are not returned.
  - **dim** (int64_t, compute input): `dim` in the example, integer on the host, specifying the dimension to apply the unique operation upon. The data type can be INT64. The value range is \[–self.dim(), self.dim()\).
  - **valueOut** (aclTensor\*, compute output): `valueOut` in the example, indicating the unique operation result, aclTensor on the device. The data type is the same as that of `self`. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas training series products</term>: The data type can be FLOAT, FLOAT16, UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, DOUBLE, or BOOL.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64, DOUBLE, BOOL, or BFLOAT16.
  - **inverseOut** (aclTensor\*, compute output): `inverseOut` in the example, aclTensor on the device, indicating the position subscript of each `self` element on the `dim` axis in **valueOut**. The data type can be INT64.
  - **countsOut** (aclTensor\*, compute output): `countsOut` in the example, aclTensor on the device, indicating the number of occurrences of each `valueOut` element in `self`. The data type can be INT64.
  - **workspaceSize** (uint64_t\*, output): size of the workspace required to be allocated on the device.
  - **executor** (aclOpExecutor\**, output): operator executor, containing the operator computation process.
- **Returns:**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, valueOut, inverseOut, or countsOut is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self is not supported.
                                        2. The data type of inverseOut or countsOut is not INT64.
                                        3. The data types of self and valueOut are inconsistent.
                                        4. The shape of self is greater than eight-dimensional.
                                        5. The dim value is not in the range of [–self.dim(), self.dim()).
  ```

## aclnnUniqueDim

- **Parameters:**
  
  - **workspace** (void\*, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnUniqueDimGetWorkspaceSize**.
  - **executor** (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - **stream** (aclrtStream, input): stream for executing the task.
- **Returns:**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
  - **aclnnUniqueDim** defaults to a deterministic implementation.
- Performance:
 	   - On A2, A3, and training series products, when the dimension value of **self** on **dim** exceeds 200 million, the performance is poor or even the running times out.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "math.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_unique_dim.h"

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
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
  // 2. Construct the input and output based on the API definition.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> valueShape = {3,2};
  std::vector<int64_t> inverseShape = {4, 2};
  std::vector<int64_t> countsShape = {3};
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
  bool sorted = false;
  bool returnInverse = false;
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

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnUniqueDim.
  ret = aclnnUniqueDimGetWorkspaceSize(self, sorted, returnInverse, dim, valueOut, inverseOut, countsOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUniqueDimGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnUniqueDim.
  ret = aclnnUniqueDim(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUniqueDim failed. ERROR: %d\n", ret); return ret);
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
