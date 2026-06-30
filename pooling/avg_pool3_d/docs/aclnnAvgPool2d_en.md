# aclnnAvgPool2d

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/pooling/avg_pool3_d)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    √     |
| <term>Atlas training series products</term>                             |    √     |

## Function

- Description: Performs two-dimensional average pooling over the input tensor with the window of $kH * kW$ and stride of $sH * sW$. Specifically, $k$ is kernelSize, indicating the size of the pooling window. $s$ is stride, indicating the stride of the pooling operation.
- Formula:
  The relationship among input ($N,C,H,W$), output out ($N,C,H_{out},W_{out}$), pooling step ($strides$), and pooling window size ($kH,kW$) is as follows:

$$
H_{out}=\lfloor \frac{H_{in}+2*paddings[0]-kH}{strides[0]}+1 \rfloor
$$

$$
W_{out}=\lfloor \frac{W_{in}+2*paddings[1]-kW}{strides[1]}+1 \rfloor
$$

$$
out(N_i,C_i,h,w)=\frac{1}{kH*kW}\sum_{m=0}^{kH-1}\sum_{n=0}^{kW-1}input(N_i,C_i,strides[0]*h+m,strides[1]*w+n)
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnAvgPool2dGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnAvgPool2d** is called to perform computation.

- `aclnnStatus aclnnAvgPool2dGetWorkspaceSize(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* strides, const aclIntArray* paddings, const bool ceilMode, const bool countIncludePad, const int64_t divisorOverride, const int8_t cubeMathType, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnAvgPool2d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnAvgPool2dGetWorkspaceSize

- **Parameters:**

  - **self** (aclTensor*, compute input): tensor to be converted ($input$ in the formula), which is a tensor on the device. Empty tensors are not supported. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported, and the [data format](../../../docs/en/context/data_formats.md) can be NCHW or NCL.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be BFLOAT16, FLOAT16, or FLOAT32.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be FLOAT16 or FLOAT32.
  - **kernelSize** (aclIntArray*, compute input): pooling window size ($k$ in the formula), which is an aclIntArray on the host. The length is 1 ($kH=kW$) or 2 ($kH, kW$). The data type is INT64. The value must be greater than 0.

  - **strides** (aclIntArray*, compute input): pooling stride ($strides$ in the formula), which is an aclIntArray on the host. The length is 1 ($sH=sW$) or 2 ($sH, sW$). The data type is INT64. The value must be greater than 0.

  - **paddings** (aclIntArray*, compute input): number of zero-padding layers in the H and W directions of the input ($paddings$ in the formula), which is an aclIntArray on the host. The length is 1 ($padH=padW$) or 2 ($padH, padW$). The data type is INT64. The value cannot be less than 0.

  - **ceilMode** (bool, compute input): whether the shape of the deduced output **out** is rounded up. The data type can be BOOL.

  - **countIncludePad** (bool, compute input): whether to include padded zeros when computing average pooling. The data type can be BOOL.

  - **divisorOverride** (int64_t, compute input): divisor for computing average pooling. The data type is INT64. If **divisorOverride** is set to **0**, the function is disabled.

  - **cubeMathType** (INT8, compute input): compute logic of the Cube unit, which is an integer on the host. The data type can be INT8. Note: If the input data types can be deduced from each other, this parameter processes the deduced data type by default. The supported enumerated values are as follows:
    * 0: KEEP_DTYPE. The input data type is retained for computation.
      * <term>Atlas training series products</term> and <term>Atlas inference series products</term>: This option is not supported when the input data type is FLOAT32.
    * 1: ALLOW_FP32_DOWN_PRECISION. The input data can be computed with reduced precision.
      * <term>Atlas training series products</term> and <term>Atlas inference series products</term>: If the input data type is FLOAT32, it will be converted to FLOAT16 for computation. When the input is of other data types, it is not processed.
      * <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: This option is not supported when the input data type is FLOAT32. When the input is of other data types, it is not processed.
    * 2: USE_FP16. The input data can be downgraded to FLOAT16 for computation.
      * <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: This option is not supported when the input data type is BFLOAT16.
    * 3: USE_HF32. This option is not supported.

  - **out** (aclTensor\*, compute output): output tensor, which is $out$ in the formula. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be NCHW or NCL. The data type and [data format](../../../docs/en/context/data_formats.md) must be the same as those of **self**.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be BFLOAT16, FLOAT16, or FLOAT32.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be FLOAT16 or FLOAT32.
  - **workSpaceSize** (uint64_t\*, output): size of the workspace to be allocated on the device.

  - **executor** (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, kernelSize, strides, paddings, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of the passed self or out is not supported.
                                    2. The data type or format of the passed self is different from that of out.
                                    3. The value of the passed self, paddings, or out is less than 0 in a dimension.
                                    4. The value of the passed kernelSize or strides is less than or equal to 0 in a dimension.
                                    5. The length of the passed kernelSize or paddings is less than 1.
                                    6. The output shape computed by average pooling is inconsistent with the specified shape.
                                    7. The padding attribute exceeds 1/2 of the position corresponding to kernelSize. For example, paddingH = 2, kernelSizeH = 2, paddingH > kernelSizeH × 1/2.
  ```

## aclnnAvgPool2d

- **Parameters:**

  - **workspace** (void \*, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API **aclnnAvgPool2dGetWorkspaceSize**.
  - **executor** (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  - **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic compute:
  - **aclnnAvgPool2d** defaults to a deterministic implementation.

- <term>Atlas training series products</term>: The Cube unit does not support FLOAT32 computation. The input data type FLOAT32 can be converted to FLOAT16 in the API for computation by setting **cubeMathType** to **1** (**ALLOW_FP32_DOWN_PRECISION**).

## Example
The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_avgpool2d.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                     \
      if (!(cond)) {                       \
          Finalize(deviceId, stream);      \
          return_expr;                     \
      }                                    \
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
  // Call aclrtMemcpy to copy the data from the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnAvgPool2dTest(int32_t deviceId, aclrtStream& stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  int64_t divisorOverride = 0;
  bool countIncludePad = true;
  bool ceilMode = false;
  int8_t cubeMathType = 2;

  std::vector<int64_t> selfShape = {1, 16, 4, 4};
  std::vector<int64_t> outShape = {1, 16, 1, 1};

  std::vector<int64_t> kernelDims = {4, 4};
  std::vector<int64_t> strideDims = {1, 1};
  std::vector<int64_t> paddingDims = {0, 0};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData(256, 2);
  std::vector<float> outHostData(16, 0);

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> selfTensorPtr(self, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create a kernel aclIntArray.
  aclIntArray *kernelSize = aclCreateIntArray(kernelDims.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> kernelSizePtr(kernelSize, aclDestroyIntArray);
  CHECK_FREE_RET(kernelSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // Create a stride aclIntArray.
  aclIntArray *strides = aclCreateIntArray(strideDims.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(strides, aclDestroyIntArray);
  CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // Create a paddings aclIntArray.
  aclIntArray *paddings = aclCreateIntArray(paddingDims.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> paddingsPtr(paddings, aclDestroyIntArray);
  CHECK_FREE_RET(paddings != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAvgPool2d.
  ret = aclnnAvgPool2dGetWorkspaceSize(self,
                                       kernelSize,
                                       strides,
                                       paddings,
                                       ceilMode,
                                       countIncludePad,
                                       divisorOverride,
                                       cubeMathType,
                                       out,
                                       &workspaceSize,
                                       &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool2dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnAvgPool2d.
  ret = aclnnAvgPool2d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool2d failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, outData[i]);
  }
  return ACL_SUCCESS;
}

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnAvgPool2dTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool2dTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
