# aclnnMaxUnpool3d

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/scatter_elements)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    ×    |
| <term>Atlas training series products</term>                             |    √    |

## Function

- Description: Performs the three-dimensional inverse operation of [aclnnMaxPool](../../../pooling/max_pool_v3/docs/aclnnMaxPool_en.md). **outputSize** determines the size of **outRef** along the D, H, and W axes, fills elements of **self** in **outRef** based on **indices**, and sets other positions to **0**.

- Formula:

  - When the input is four-dimensional, the dimensions are N, D, H, and W, respectively. N indicates the batch size, H indicates the feature map height, W indicates the feature map width, and D indicates the feature map depth.
  
    $$
    outRef[N][indices[N][i]] = self[N][i]
    $$

  - When the input is five-dimensional, the dimensions are N, C, D, H, and W, respectively. C indicates the number of feature map channels.
    
    $$
    outRef[N][C][indices[N][C][i]] = self[N][C][i]
    $$
    
  
    **outRef**, **indices**, and **self** are obtained by reshaping the axis combined from the last two axes, that is, i ∈ [0, D * H * W).

## Prototype

- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnMaxUnpool3dGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnMaxUnpool3d** is called to perform computation.

  - `aclnnStatus aclnnMaxUnpool3dGetWorkspaceSize(const aclTensor* self, const aclTensor* indices, const aclIntArray* outputSize, const aclIntArray* stride, const aclIntArray* padding, aclTensor* outRef, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnMaxUnpool3d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMaxUnpool3dGetWorkspaceSize

- **Parameters:**
  
  - **self** (aclTensor\*, compute input): target tensor to be converted, `self` in the formula, aclTensor on the device. The data type can be FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT8, or DOUBLE, and is the same as that of **outRef**. The shape is the same as that of **indices**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. The tensor can be four-dimensional (N, D, H, W) or five-dimensional (N, C, D, H, W).
  - **indices** (aclTensor\*, compute input): indexes of the input **self** elements in the output result, `indices` in the formula, aclTensor on the device. The data type can be INT64 or INT32, and the shape must be the same as that of **self**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. The tensor can be four-dimensional (N, D, H, W) or five-dimensional (N, C, D, H, W).
  - **outputSize** (aclIntArray\*, compute input): aclIntArray on the host. The size is 3. The product of three elements must be greater than or equal to the product of the sizes of **self** in the D, H, and W dimensions. This parameter indicates the spatial size of output in the D, H, and W dimensions.
  - **stride** (aclIntArray\*, compute input): reserved. This parameter does not participate in computation in the current version. It is an aclIntArray on the host, with the size of 3 and the value greater than 0. This parameter indicates the step size of the maximum pooling window in the D, H, and W dimensions.
  - **padding** (aclIntArray\*, compute input): reserved. This parameter does not participate in computation in the current version. It is an aclIntArray on the host, with the size of 3. This parameter indicates the padding value of the maximum pooling window in the D, H, and W dimensions.
  - **outRef** (aclTensor\*, compute output): output `outRef` in the formula, aclTensor on the device. The data type can be FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT8, or DOUBLE, and must be the same as that of **self**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. The tensor can be four-dimensional (N, D, H, W) or five-dimensional (N, C, D, H, W).
  - **workspaceSize** (uint64_t\*, output): size of the workspace to be allocated on the device.
  - **executor** (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.
  
- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, indices, outputSize, stride, padding, or outRef is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self or indices is not supported.
                                        2. The data types of self and outRef are inconsistent.
                                        3. The dimension of self is not four or five.
                                        4. The shape of self is different from that of indices.
                                        5. The size of outputSize is not 3.
                                        6. The product of three elements of outputSize is less than the product of sizes of self in the D, H, and W dimensions.
                                        7. The size of stride is not 3.
                                        8. The size of padding is not 3.
                                        9. The size of self in the C, D, H, and W dimensions is not greater than 0.
                                        10. The element values of stride are not greater than 0.
                                        11. The sizes of outRef in the N and C dimensions are not completely the same as those of self.
                                        12. The sizes of outRef in the D, H, and W dimensions are different from the values of the three elements in outputSize.
  ```

## aclnnMaxUnpool3d

- **Parameters:**
  
  - **workspace** (void\*, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnMaxUnpool3dGetWorkspaceSize**.
  - **executor** (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - **stream** (aclrtStream, input): stream for executing the task.
  
- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
  - **aclnnMaxUnpool3d** defaults to a deterministic implementation.

## Example
The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_unpool3d.h"

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
  std::vector<int64_t> selfShape = {1, 1, 2, 2};
  std::vector<int64_t> outShape = {1, 1, 4, 4};
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* indices = nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<float> outHostData = {0, 0, 0, 0.0, 0, 0, 0, 0,
                                    0, 0, 0, 0.0, 0, 0, 0, 0};
  std::vector<int64_t> indicesHostData = {3, 8, 11, 13};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an indices aclTensor.
  ret = CreateAclTensor(indicesHostData, selfShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> arraySize1 = {1, 4, 4};
  const aclIntArray *outputSize = aclCreateIntArray(arraySize1.data(), arraySize1.size());
  CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  std::vector<int64_t> arraySize2 = {1, 2, 3};
  const aclIntArray *stride = aclCreateIntArray(arraySize2.data(), arraySize2.size());
  CHECK_RET(stride != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  const aclIntArray *padding = aclCreateIntArray(arraySize2.data(), arraySize2.size());
  CHECK_RET(padding != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnMaxUnpool3d.
  ret = aclnnMaxUnpool3dGetWorkspaceSize(self, indices, outputSize, stride, padding, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool3dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMaxUnpool3d.
  ret = aclnnMaxUnpool3d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool3d failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out[%ld] is: %f\n", i, outData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(indices);
  aclDestroyIntArray(outputSize);
  aclDestroyIntArray(stride);
  aclDestroyIntArray(padding);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(indicesDeviceAddr);
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
