# aclnnAdaptiveMaxPool2dBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/pooling/adaptive_max_pool3d_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    ×     |

## Function

- Description:
  Performs backpropagation of forward adaptive max pooling. The gradient is backfilled to the coordinate of the maximum value in each adaptive window. The coordinates are accumulated.
- Forward propagation formula:
  For the input self with shape $[N, C, H, W]$, where N indicates the batch size, C indicates the number of feature map channels, H indicates the feature map height, and W indicates the feature map width. In the scenario where **outputSize** is **$[H_o, W_o]$**, the output shape is $[N, C, H_o, W_o]$ and the shape of **indices** is $[N, C, H_o, W_o]$. The calculation formula for each element in the corresponding tensor is as follows:
  
  $$
  H_{left}^m = \lfloor(m*H)/H_o\rfloor \\
  H_{right}^m = \lceil(m*H)/H_o\rceil  \\
  W_{left}^n = \lfloor(n*W)/W_o\rfloor \\
  W_{right}^n = \lceil(n*W)/W_o\rceil  \\
  output(N,C,l,m,n) = \mathop{\max}\limits_{i \in ,j\in[H_{left}^m,H_{right}^m],k\in[W_{left}^n,W_{right}^n]} input(N,C,i,j,k) \\
  indices(N,C,l,m,n) = \mathop{\arg\max}\limits_{i \in ,j\in[H_{left}^m,H_{right}^m],k\in[W_{left}^n,W_{right}^n]} input(N,C,i,j,k)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnAdaptiveMaxPool2dBackward** is called to perform computation.

- `aclnnStatus aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnAdaptiveMaxPool2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize

- **Parameters:**
  
  - **gradOutput** (aclTensor \*, compute input): gradient tensor, which is an aclTensor on the device. The data type can be FLOAT32, FLOAT16, or BFLOAT16, and the shape is the same as that of the forward output. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. When the input is 4D, it is processed as NCHW. When the input is 3D, it is padded with 1s in dimension 0 and processed as NCHW.
    
  - **self** (aclTensor \*, compute input): forward input tensor, which is an aclTensor on the device. The data type can be FLOAT32, FLOAT16, or BFLOAT16. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. When the input is 4D, it is processed as NCHW. When the input is 3D, it is padded with 1s in dimension 0 and processed as NCHW, same as that of **gradOutput**.
    
  - **indices** (aclTensor \*, compute input): input tensor, which is an aclTensor on the device. The data type can be INT32 or INT64. It indicates the index of the maximum element in the forward input. The [data format](../../../docs/en/context/data_formats.md) must be the same as that of **gradOutput**. The shape is the same as that of **gradOutput**.
    
  - **gradInput** (aclTensor \*, compute output): reverse output tensor, which is an aclTensor on the device. The data type can be FLOAT32, FLOAT16, or BFLOAT16, and the shape is same as that of **self**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) must be the same as that of **self**.
    
  - **workspaceSize** (uint64_t \*, output): size of the workspace to be allocated on the device.
  - **executor** (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.
- **Returns:**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  
  ```
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed gradOutput, self, or indices is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of gradOutput, self, indices, or gradInput is not supported.
                                   2. The data format of gradOutput, self, indices, or gradInput is not supported.
                                   3. The input and output shapes are not 3D or 4D.
                                   4. The shapes of gradOutput and indices are inconsistent, and the shapes of self and gradInput are inconsistent.
                                   5. height × width > max int32, which exceeds the expression range of indices.
  ```

## aclnnAdaptiveMaxPool2dBackward

- **Parameters:**
  
  - **workspace** (void \*, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API **aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize**.
  - **executor** (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  - **stream** (aclrtStream, input): stream for executing the task.
- **Returns:**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic compute:
  - **aclnnAdaptiveMaxPool2dBackward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

- In the case of non-integer division (the remainder of the dimension values of the last two dimensions of **self** divided by the dimension values of the last two dimensions of **gradOutput** is not 0), the shape cannot exceed 2 to the power of 24. If the shape is exactly divisible, there is no such restriction.
- Example of integer division: self = [a, b, c, d], gradOutput = [a, b, e, f]. If c%e is 0 and d%f is 0, it is an integer division scenario.
## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_adaptive_max_pool2d_backward.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> gradOutShape = {1, 1, 1, 1};
  std::vector<int64_t> selfShape = {1, 1, 2, 2};
  std::vector<int64_t> indicesShape = {1, 1, 1, 1};
  std::vector<int64_t> gradInShape = {1, 1, 2, 2};
  void* gradOutDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* gradInDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* self = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* gradIn = nullptr;
  std::vector<float> gradOutHostData = {0.4757};
  std::vector<float> selfHostData = {0.0850, -0.5147, -0.0212, -0.5654};
  std::vector<int32_t> indicesHostData = {6};
  std::vector<float> gradInHostData = {0, 0, 0, 0};

  // Create a gradOut aclTensor.
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gradIn aclTensor.
  ret = CreateAclTensor(gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT, &gradIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnAdaptiveMaxPool2dBackward API call example
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  // Call the first-phase API aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize.
  ret = aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize(gradOut, self, indices, gradIn, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnAdaptiveMaxPool2dBackward.
  ret = aclnnAdaptiveMaxPool2dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveMaxPool2dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(gradInShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradIn result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(gradOut);
  aclDestroyTensor(self);
  aclDestroyTensor(indices);
  aclDestroyTensor(gradIn);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(gradInDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
