# aclnnScatterValue&aclnnInplaceScatterValue

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/scatter_elements_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    √     |

## Function

- Description: Writes values from scalar **value** into tensor **self** one by one according to the specified axis, direction, and corresponding position relationship. **value** will be broadcast into tensor **src** whose shape is the same as that of tensor **index** for Scatter computation.

- **Example:**
  For a three-dimensional tensor, **self** is updated based on the following rules:

  ```
  self[index[i][j][k]][j][k] = value # if dim == 0
  self[i][index[i][j][k]][k] = value # if dim == 1
  self[i][j][index[i][j][k]] = value # if dim == 2
  ```

  The following conditions must be met during computation:
  - **self** and **index** must have the same number of dimensions.
  - For each dimension d, if d != dim, there is a restriction that index.size(d) <= self.size(d).
  - The value of dim must be in the range of [–(number of dimensions of self), (number of dimensions of self) – 1].
  - The number of dimensions of **self** must be less than or equal to eight.
  - The corresponding dim value in index must be in the range of [0, self.size(dim) – 1].

## Prototype

- **aclnnScatterValue** and **aclnnInplaceScatterValue** implement the same function in different ways. Select a proper operator based on your requirements.

  - **aclnnScatterValue**: An output tensor object needs to be created to store the computation result.
  - **aclnnInplaceScatterValue**: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnScatterValueGetWorkspaceSize** or **aclnnInplaceScatterValueGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnScatterValue** or **aclnnInplaceScatterValue** is called to perform computation.

  - `aclnnStatus aclnnScatterValueGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclScalar *value, int64_t reduce, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnScatterValue(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`
  - `aclnnStatus aclnnInplaceScatterValueGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclTensor *index, const aclScalar *value, int64_t reduce, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnInplaceScatterValue(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnScatterValueGetWorkspaceSize
- **Parameters:**

  - **self** (aclTensor*, compute input): `self` in the formula, aclTensor on the device. It indicates the target tensor of scatter. **self** must have the same number of dimensions as **index**, and the shape supports zero to eight dimensions. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas training series products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, or COMPLEX128.
    - <term>Atlas A2 training series products, Atlas A2 inference series products</term>, <term>Atlas A3 training series products, and Atlas A3 inference series products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, or BFLOAT16.

  - **dim** (int64_t, compute input): dimension used for scattering. The data type can be INT64. The value range is [–(number of dimensions of self), (number of dimensions of self) – 1].

  - **index** (aclTensor*, compute input): `index` in the formula, aclTensor on the device. It indicates the index tensor. The data type can be INT32 or INT64. **index** must have the same number of dimensions as **self**, and the shape supports zero to eight dimensions. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.

  - **value** (aclScalar*, compute input): aclScalar on the host. When **value** is **COMPLEX**, **self** must also be a **COMPLEX** tensor. There are no other data type restrictions.

  - **reduce** (int64_t, compute input): the reduction operation to be applied. The options and corresponding int values are add (1), mul (2), and none (0). The specific operations are defined as follows:
    **0**: replaces **value** to the corresponding position in **out** according to **index**.
    **1**: accumulates **value** to the corresponding position in **out** according to **index**.
    **2**: multiplies **value** to the corresponding position in **out** according to **index**.
  - **out** (aclTensor*, compute output): tensor for storing the scatter output. The [data format](../../../docs/en/context/data_formats.md), data type, and shape must be the same as those of **self**.
    - <term>Atlas training series products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, or COMPLEX128.
    - <term>Atlas A2 training series products, Atlas A2 inference series products</term>, <term>Atlas A3 training series products, and Atlas A3 inference series products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, or BFLOAT16.

  - **workspaceSize**(uint64_t*, output parameter): size of the workspace required to be allocated on the device.

  - **executor**(aclOpExecutor, output parameter): operator executor, containing the operator computation process.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, index, value, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, index, value, or out is not supported.
                                        2. The data types of self and out are inconsistent.
                                        3. The numbers of dimensions of self and index are inconsistent.
                                        4. The shapes of self and out are inconsistent.
                                        5. The shape of self or index does not meet the following requirements:
                                          For each dimension d, if d != dim, there is a restriction that index.size(d) <= self.size(d).
                                        6. The value of dim is not in the range of [–(number of dimensions of self), (number of dimensions of self) – 1].
                                        7. The number of dimensions of self is greater than eight.
                                        8. When value is COMPLEX, the data type of self is not COMPLEX.
  ```

## aclnnScatterValue

- **Parameters:**

  - **workspace**(void*, input parameter): address of the workspace to be allocated on the device.

  - **workspaceSize**(uint64_t, input parameter): size of the workspace to be allocated on the device, which is obtained by calling the first-phase **aclnnScatterValueGetWorkspaceSize**.

  - **executor**(aclOpExecutor*, input parameter): operator executor, containing the operator computation process.

  - **stream**(aclrtStream, input parameter): stream for executing the task.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## aclnnInplaceScatterValueGetWorkspaceSize

- **Parameters:**

  - **selfRef** (aclTensor*, compute input | compute output): `self` in the formula, aclTensor on the device. It indicates the target tensor of scatter. **selfRef** must have the same number of dimensions as **index**, and the shape supports zero to eight dimensions. Empty tensors and [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas training series products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT, DOUBLE, COMPLEX64, or COMPLEX128.
    - <term>Atlas A2 training series products, Atlas A2 inference series products</term>, <term>Atlas A3 training series products, and Atlas A3 inference series products</term>: The data type can be UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT, DOUBLE, COMPLEX64, COMPLEX128, or BFLOAT16.
  - **dim** (int64_t, compute input): dimension used for scattering. The data type can be INT64. The value range is [–(number of dimensions of selfRef), (number of dimensions of selfRef) – 1].
  - **index** (aclTensor*, compute input): `index` in the formula, aclTensor on the device. It indicates the index tensor. The data type can be INT32 or INT64. **index** must have the same number of dimensions as **selfRef**, and the shape supports zero to eight dimensions. For each dimension d, if d != dim, ensure that index.size(d) <= selfRef.size(d). Empty tensors and [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **value** (aclScalar*, compute input): When **value** is **COMPLEX**, **selfRef** must also be a **COMPLEX** tensor. There are no other data type restrictions.
  - **reduce** (int64_t, compute input): the reduction operation to be applied. The options and corresponding int values are add (1), mul (2), and none (0). The specific operations are defined as follows:
    **0**: replaces **value** to the corresponding position in **selfRef** according to **index**.
    **1**: accumulates **value** to the corresponding position in **selfRef** according to **index**.
    **2**: multiplies **value** to the corresponding position in **selfRef** according to **index**.
  - **workspaceSize**(uint64_t, output parameter): size of the workspace required to be allocated on the device.
  - **executor**(aclOpExecutor**, output parameter): operator executor, containing the operator computation process.

- **Returns:**
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed selfRef, index, or value is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of selfRef, index, or value is not supported.
                                        2. The data types of selfRef and out are inconsistent.
                                        3. The numbers of dimensions of selfRef and index are inconsistent.
                                        4. The shapes of selfRef and out are inconsistent.
                                        5. The shape of selfRef or index does not meet the following requirements:
                                         For each dimension d, if d != dim, there is a restriction that index.size(d) <= selfRef.size(d).
                                        6. The value of dim is not in the range of [–(number of dimensions of selfRef), (number of dimensions of selfRef) – 1].
                                        7. The number of dimensions of selfRef is greater than eight.
                                        8. When value is COMPLEX, the data type of selfRef is not COMPLEX.
  ```

## aclnnInplaceScatterValue

- **Parameters:**
  - **workspace**(void*, input parameter): address of the workspace to be allocated on the device.
  - **workspaceSize**(uint64_t, input parameter): size of the workspace to be allocated on the device, which is obtained by calling the first-phase **aclnnInplaceScatterValueGetWorkspaceSize**.
  - **executor**(aclOpExecutor*, input parameter): operator executor, containing the operator computation process.
  - **stream**(aclrtStream, input parameter): stream for executing the task.

- **Returns:**
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
  - **aclnnScatterValue** and **aclnnInplaceScatterValue** default to a deterministic implementation.


## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

**aclnnScatterValue sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter.h"

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
  int64_t dim = 1;
  int64_t reduce = 1;
  std::vector<int64_t> selfShape = {3, 4};
  std::vector<int64_t> indexShape = {2, 3};
  std::vector<int64_t> outShape = {3, 4};
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclScalar* value = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int64_t> indexHostData = {0, 0, 2, 1, 0, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float Value = 1.2f;

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an index aclTensor.
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a value aclScalar.
  value = aclCreateScalar(&Value, aclDataType::ACL_FLOAT);
  CHECK_RET(value != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnScatterValue.
  ret = aclnnScatterValueGetWorkspaceSize(self, dim, index, value, reduce, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterValueGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnScatterValue.
  ret = aclnnScatterValue(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterValue failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(index);
  aclDestroyScalar(value);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(indexDeviceAddr);
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

**aclnnInplaceScatterValue sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter.h"

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
  int64_t dim = 1;
  int64_t reduce = 1;
  std::vector<int64_t> selfRefShape = {3, 4};
  std::vector<int64_t> indexShape = {2, 3};
  void* selfRefDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* index = nullptr;
  aclScalar* value = nullptr;
  std::vector<float> selfRefHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int64_t> indexHostData = {0, 0, 2, 1, 0, 2};
  float Value = 1.2f;

  // Create a selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an index aclTensor.
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a value aclScalar.
  value = aclCreateScalar(&Value, aclDataType::ACL_FLOAT);
  CHECK_RET(value != nullptr, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceScatterValue.
  ret = aclnnInplaceScatterValueGetWorkspaceSize(selfRef, dim, index, value, reduce, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterValueGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceScatterValue.
  ret = aclnnInplaceScatterValue(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterValue failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(index);
  aclDestroyScalar(value);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(indexDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
