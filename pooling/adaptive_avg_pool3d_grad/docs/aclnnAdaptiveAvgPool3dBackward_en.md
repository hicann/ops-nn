# aclnnAdaptiveAvgPool3dBackward
## Supported Products
[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/pooling/adaptive_avg_pool3d_grad)


| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    ×     |

## Function

Description: Performs backpropagation of [aclnnAdaptiveAvgPool3d](../../adaptive_avg_pool3d/docs/aclnnAdaptiveAvgPool3d_en.md).

## Prototype
Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnAdaptiveAvgPool3dBackward** is called to perform computation.

- `aclnnStatus aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnAdaptiveAvgPool3dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize

- **Parameters:**
  
  - **gradOutput** (aclTensor*, compute input): gradient of the current node, aclTensor on the device. The data type can be BFLOAT16, FLOAT16, or FLOAT32, and must be the same as that of **self**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The shape can be 4D or 5D. The dimension values are positive numbers, and the total number of dimensions is the same as that of **self**. The [data format](../../../docs/en/context/data_formats.md) can be NCDHW or ND, and must be the same as that of **self**.
  - **self** (aclTensor\*, compute input): input tensor, leaf node, aclTensor on the device. The data type can be BFLOAT16, FLOAT16, or FLOAT32. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The shape can be 4D or 5D, and the dimension values are positive numbers. The [data format](../../../docs/en/context/data_formats.md) can be NCDHW or ND.
  - **out** (aclTensor\*, compute output): output tensor, corresponding to the gradient of the input leaf node, aclTensor on the device. The data type can be BFLOAT16, FLOAT16, or FLOAT32. The shape must be the same as that of **self**. The [data format](../../../docs/en/context/data_formats.md) can be NCDHW or ND, and the data type must be the same as that of **self**.
  - **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.
  - **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.
- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed gradOutput, self, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of gradOutput or self is not supported.
                                        2. The data types of gradOutput, self, and out are inconsistent.
                                        3. The shape of gradOutput, self, and out is not 4D or 5D.
                                        4. The shapes of gradOutput, self, and out do not match.
                                        5. The shape of gradOutput or self has a dimension not greater than 0.
                                        6. The data formats of gradOutput and self are inconsistent.
                                        7. The N and C dimensions are inconsistent.
  ```

## aclnnAdaptiveAvgPool3dBackward

- **Parameters:**
  
  - **workspace** (void*, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API **aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize**.
  - **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - **stream** (aclrtStream, input): stream for executing the task.
- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic compute:
  - **aclnnAdaptiveAvgPool3dBackward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "math.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_adaptive_avg_pool3d_backward.h"

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
  // Call aclrtMemcpy to copy the data from the host to the device.
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
  // 1. Initialize the device and stream. For details, see the ACL API manual.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output.
  std::vector<int64_t> yGradShape = {2, 2, 1, 1, 2};
  std::vector<int64_t> xShape = {2, 2, 1, 1, 4};
  std::vector<int64_t> xGradShape = {2, 2, 1, 1, 4};
  void* yGradDeviceAddr = nullptr;
  void* xDeviceAddr = nullptr;
  void* xGradDeviceAddr = nullptr;
  aclTensor* yGrad = nullptr;
  aclTensor* x = nullptr;
  aclTensor* xGrad = nullptr;
  std::vector<float> yGradHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> xHostData(GetShapeSize(xShape), 1);
  std::vector<float> xGradHostData(16, 0);
  // Create a yGrad aclTensor.
  ret = CreateAclTensor(yGradHostData, yGradShape, &yGradDeviceAddr, aclDataType::ACL_FLOAT, &yGrad);
  // Create an x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an xGrad aclTensor.
  ret = CreateAclTensor(xGradHostData, xGradShape, &xGradDeviceAddr, aclDataType::ACL_FLOAT, &xGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAdaptiveAvgPool3dBackward.
  ret = aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize(yGrad, x, xGrad, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAdaptiveAvgPool3dBackward.
  ret = aclnnAdaptiveAvgPool3dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveAvgPool3dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host.
  auto size = GetShapeSize(xGradShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), xGradDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor.
  aclDestroyTensor(yGrad);
  aclDestroyTensor(x);
  aclDestroyTensor(xGrad);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(yGradDeviceAddr);
  aclrtFree(xDeviceAddr);
  aclrtFree(xGradDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
