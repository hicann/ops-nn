# aclnnLinalgVectorNorm

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/lp_norm_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    √    |
| <term>Atlas inference series products</term>                            |    ×    |
| <term>Atlas training series products</term>                             |    √    |

## Function

- Description: Computes the vector norm of the input tensor.
- Formula: **n** indicates the size of the dimensions specified by **dims**.

$$
out=(\sum_{i=1}^{n}|self_{i}|^{ord})^{1/ord}
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnLinalgVectorNormGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnLinalgVectorNorm** is called to perform computation.

- `aclnnStatus aclnnLinalgVectorNormGetWorkspaceSize(const aclTensor* self, const aclScalar* ord, const aclIntArray* dims, bool keepDims, const aclDataType dtype, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnLinalgVectorNorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnLinalgVectorNormGetWorkspaceSize

- **Parameters:**

  - **self** (aclTensor*, compute input): `self` in the formula, aclTensor on the device. The shape supports 0 to 8 dimensions. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported, and the [data format](../../../docs/en/context/data_formats.md) supports ND.
    - <term>Atlas training series products</term> and <term>Atlas 200I/500 A2 inference products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.

  - **ord** (aclScalar*, compute input): norm type, `ord` in the formula, and aclScalar on the device.
    - <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas training series products</term>, <term>Atlas A2 training series products/Atlas A2 inference series products</term>, and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The value can be 0.0, 1.0, 2.0, or 3.0. The data type can be FLOAT.
  - **dims** (aclIntArray*, compute input): dimension for computing the norm, aclIntArray on the device. The value range is [-N, N-1]. The elements in **dims** must be unique. **N** indicates the dimension of **self**. The default value is **None**, in which case **self** is flattened and then calculated. The data type can be INT64.

  - **keepDims** (bool, compute input): whether to retain the axis specified by **dims** in the output tensor. It is a bool constant on the host.

  - **dtype** (aclDataType, compute input): data type for **self** computation, aclDataType constant on the host. The data type must be the same as that of **out**. Before computation, convert **self** to the specified data type. The data types of **dtype** and **self** must meet the conversion relationship: When the data type of **self** is FLOAT, **dtype** can only be **FLOAT**. When the data type of **self** is FLOAT16, **dtype** can be FLOAT or FLOAT16. When the data type of **self** is BFLOAT16, **dtype** can be FLOAT or BFLOAT16.
    - <term>Atlas training series products</term> and <term>Atlas 200I/500 A2 inference products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.

  - **out** (aclTensor*, compute output): `out` in the formula, aclTensor on the device. The shape supports 0 to 8 dimensions. If **keepDims** is **true**, the shape of dimensions must be the same as that of **self** except that the size of the dimension specified by **dim** is 1. If **keepDims** is **false**, the dimension of the reduce axis is not reserved, and the shape of other dimensions must be the same as that of **self**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported, and the [data format](../../../docs/en/context/data_formats.md) supports ND.
    - <term>Atlas training series products</term> and <term>Atlas 200I/500 A2 inference products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, or BFLOAT16.

  - **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.

  - **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): The passed self, ord, dims, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self or out is not supported.
                                        2. The value of dims exceeds the range [-N, N-1], where N indicates the dimensions of self.
                                        3. The data types of dtype and out are different.
                                        4. Duplicate values exist in dims.
                                        5. The value of ord is not supported.
                                        6. The shape of self or out exceeds eight dimensions.
                                        7. The shape of out is not equal to the shape deduced from self, dim, and keepDim.
                                        8. The dtype does not meet the conversion relationship with the data type of self.
  ```

## aclnnLinalgVectorNorm

- **Parameters:**

  - **workspace** (void*, input): address of the workspace to be allocated on the device.

  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnLinalgVectorNormGetWorkspaceSize**.

  - **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - **stream** (aclrtStream, input): stream for executing the task.


- **Returns:**
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).


## Constraints

- Deterministic compute
  - **aclnnLinalgVectorNorm** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_linalg_vector_norm.h"

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
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {1, 2};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclScalar* ord = nullptr;
  aclIntArray* dims = nullptr;

  std::vector<float> selfHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};
  std::vector<float> outHostData = {0.0, 0.0};
  std::vector<int64_t> dimData = {0};
  float pValue = 2.0f;
  bool keepDims = true;
  aclDataType dtype = aclDataType::ACL_FLOAT;

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an ord aclScalar.
  ord = aclCreateScalar(&pValue, aclDataType::ACL_FLOAT);
  CHECK_RET(ord != nullptr, return ret);
  // Create a dim aclIntArray.
  dims = aclCreateIntArray(dimData.data(), 1);
  CHECK_RET(dims != nullptr, return ret);

  // 3. Call the CANN operator library API. Modify the API as required.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnLinalgVectorNorm.
  ret = aclnnLinalgVectorNormGetWorkspaceSize(self, ord, dims, keepDims, dtype, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinalgVectorNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnLinalgVectorNorm.
  ret = aclnnLinalgVectorNorm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinalgVectorNorm failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyIntArray(dims);
  aclDestroyScalar(ord);
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
