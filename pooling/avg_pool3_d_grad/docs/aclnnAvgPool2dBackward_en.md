# aclnnAvgPool2dBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/pooling/avg_pool3_d_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    √     |
| <term>Atlas training series products</term>                             |    √     |

## Function

- Description: Performs backpropagation of 2D average pooling, which calculates the input gradient of 2D average pooling forward propagation.

- The calculation formula is as follows: Assume that the input tensor of forward propagation of 2D average pooling is $X$, the output tensor is $Y$, the pooling window size is $k*k$, and the stride is $s$. The gradient $\frac{\partial L}{\partial X}$ of $X$ is calculated as follows:

  $$
  \frac{\partial L}{\partial X_{i,j}}=\frac{1}{k^2}\sum_{n=0}^{k-1}\frac{\partial L}{\partial Y_{\lfloor\frac{i*s+m}{k}\rfloor,\lfloor\frac{j*s+n}{k}\rfloor}}
  $$

  The parameters are described as follows:

  * $L$ is the loss function, and $\lfloor\cdot\rfloor$ indicates rounding up.
  * $X_{i,j}$ indicates the $i$th row and $j$th column of the input feature map.
  * $Y_{\lfloor\frac{i*s+m}{k}\rfloor,\lfloor\frac{j*s+n}{k}\rfloor}$ indicates the pixel value in row $\lfloor\frac{i*s+m}{k}\rfloor$ and column $\lfloor\frac{j*s+n}{k}\rfloor$ of the output feature map.
  * $k$ indicates the size of the pooling window.
  * $s$ indicates the stride.
  * $\frac{\partial L}{\partial X_{i,j}}$ indicates the partial derivative of the loss function L with respect to the pixel value in the ith row and jth column of the input feature map.
  * $\frac{\partial L}{\partial Y_{\lfloor\frac{i*s+m}{k}\rfloor,\lfloor\frac{j*s+n}{k}\rfloor}}$ indicates the partial derivative of the loss function $L$ with respect to the pixel value in row $\lfloor\frac{is+m}{k}\rfloor$ and column $\lfloor\frac{js+n}{k}\rfloor$ of the feature map.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnAvgPool2dBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnAvgPool2dBackward** is called to perform computation.

- `aclnnStatus aclnnAvgPool2dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding, bool ceilMode, bool countIncludePad, int64_t divisorOverride, int8_t cubeMathType, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnAvgPool2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnAvgPool2dBackwardGetWorkspaceSize

- **Parameters:**
  - **gradOutput** (aclTensor *, compute input): input gradient, which is $\frac{\partial L}{\partial y}$ in the formula. It is an aclTensor on the device and does not support empty tensors. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be NCHW or NCL.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be BFLOAT16, FLOAT16, or FLOAT32.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be FLOAT16 or FLOAT32.
  - **self** (aclTensor *, compute input): input data, which is an aclTensor on the device and indicates the input in the forward process. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be NCHW or NCL.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be BFLOAT16, FLOAT16, or FLOAT32.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be FLOAT16 or FLOAT32.
  - **kernelSize** (aclIntArray *, compute input): pooling window size ($k$ in the formula), which is an aclIntArray on the host. It is an INT64 array, with a length of 1 ($kH=kW$) or 2 ($kH, kW$).
  - **stride** (aclIntArray *, compute input): pooling stride ($strides$ in the formula), which is an aclIntArray on the host. It is an INT64 array, with a length of 0 ($sH=kH,sW=kW$), 1 ($sH=sW$), or 2 ($sH, sW$).
  - **padding** (aclIntArray *, compute input): number of zero-padding layers in the H and W directions of the input ($paddings$ in the formula), which is an aclIntArray on the host. It is an INT64 array, with a length of 1 ($padH=padW$) or 2 ($padH, padW$).
  - **ceilMode** (bool, compute input): whether to round up the output shape deduced during forward average pooling. The data type can be BOOL.
  - **countIncludePad** (bool, compute input): whether to include the padded zeros when computing forward average pooling. The data type can be BOOL.
  - **divisorOverride** (int64_t, compute input): divisor for averaging.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The value range is not limited. The value **0** indicates that whether padding is involved in average computation is not affected. The data type can be INT64.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The value range is [–255, +255]. The value **0** indicates that whether padding is involved in average computation is not affected. The data type can be INT64.
  - **cubeMathType** (INT8, compute input): compute logic of the Cube unit, which is an integer on the host. The data type can be INT8. Note: If the input data types can be deduced from each other, this parameter processes the deduced data type by default. The supported enumerated values are as follows:
    * 0: KEEP_DTYPE. The input data type is retained for computation.
      * <term>Atlas training series products </term> and <term>Atlas inference series products</term>: This option is not supported when the input data type is FLOAT32.
    * 1: ALLOW_FP32_DOWN_PRECISION. The input data can be computed with reduced precision.
      * <term>Atlas training series products</term> and <term>Atlas inference series products</term>: If the input data type is FLOAT32, it will be converted to FLOAT16 for computation. When the input is of other data types, it is not processed.
      * <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: If the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, it is not processed.
    * 2: USE_FP16. The input data can be downgraded to FLOAT16 for computation.
      * <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: This option is not supported when the input data type is BFLOAT16.
    * 3: USE_HF32. The input data can be downgraded to HFLOAT32 for computation.
      * <term>Atlas training series products</term> and <term>Atlas inference series products</term>: This option is not supported.
      * <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: If the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, this option is not supported.
  - **gradInput** (aclTensor *, compute output): aclTensor on the device. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be NCHW or NCL. The data type and [data format](../../../docs/en/context/data_formats.md) must be the same as those of **self**.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be BFLOAT16, FLOAT16, or FLOAT32.
    - <term>Atlas inference series products</term> and <term>Atlas training series products</term>: The data type can be FLOAT16 or FLOAT32.
  - **workSpaceSize** (uint64_t \*, output): size of the workspace to be allocated on the device.
  - **executor** (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed gradOutput, self, kernelSize, padding, or gradInput is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of gradOutput or gradInput is not supported.
                                    2. The data type or format of self is inconsistent with that of gradInput.
                                    3. The dimensions of gradOutput, self, kernelSize, stride, or gradInput are not supported.
                                    4. The value of a dimension of gradOutput, self, kernelSize, stride, or gradInput is less than 0.
                                    5. The passed cubeMathType is not supported.
                                    6. The padding attribute exceeds 1/2 of the position corresponding to kernelSize. For example, paddingH = 2, kernelSizeH = 2, paddingH > kernelSizeH × 1/2.
                                    7. The passed divisorOverride is not supported.
  ```

## aclnnAvgPool2dBackward

- **Parameters:**

  - **workspace** (void\*, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnAvgPool2dBackwardGetWorkspaceSize.
  - **executor** (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic compute:
  - **aclnnAvgPool2dBackward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

- <term>Atlas training series products</term> and <term>Atlas inference series products</term>: The Cube unit does not support FLOAT32 computation. The input data type FLOAT32 can be converted to FLOAT16 in the API for computation by setting **cubeMathType** to **1** (**ALLOW_FP32_DOWN_PRECISION**).

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_avgpool2d_backward.h"

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
                    aclDataType dataType, aclTensor** tensor, aclFormat Format = aclFormat::ACL_FORMAT_ND) {
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, Format, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> gradOutputShape = {1, 16, 1, 1};
  std::vector<int64_t> selfShape = {1, 16, 4, 4};
  std::vector<int64_t> kernelDims = {4, 4};
  std::vector<int64_t> strideDims = {1, 1};
  std::vector<int64_t> paddingDims = {0, 0};
  bool ceilMode = false;
  int64_t divisorOverride = 0;
  bool countIncludePad = true;
  int8_t cubeMathType = 1;
  std::vector<int64_t> gradInputShape = {1, 16, 4, 4};

  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;

  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* gradInput = nullptr;

  std::vector<float> gradOutputHostData(GetShapeSize(gradOutputShape) * 2, 1);
  std::vector<float> selfHostData(GetShapeSize(selfShape) * 2, 1);
  std::vector<float> gradInputHostData(GetShapeSize(gradInputShape) * 2, 1);

  // Create a gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput, aclFormat::ACL_FORMAT_NCHW);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self, aclFormat::ACL_FORMAT_NCHW);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a gradInput aclTensor.
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput, aclFormat::ACL_FORMAT_NCHW);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a kernel aclIntArray.
  aclIntArray *kernelSize = aclCreateIntArray(kernelDims.data(), 2);

  // Create a stride aclIntArray.
  aclIntArray *stride = aclCreateIntArray(strideDims.data(), 2);

  // Create a paddings aclIntArray.
  aclIntArray *padding = aclCreateIntArray(paddingDims.data(), 2);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAvgPool2dBackward.
  ret = aclnnAvgPool2dBackwardGetWorkspaceSize(gradOutput,
                                       self,
                                       kernelSize,
                                       stride,
                                       padding,
                                       ceilMode,
                                       countIncludePad,
                                       divisorOverride,
                                       cubeMathType,
                                       gradInput,
                                       &workspaceSize,
                                       &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAvgPool2dBackward.
  ret = aclnnAvgPool2dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool2dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), gradInputDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, outData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(gradInput);
  aclDestroyIntArray(kernelSize);
  aclDestroyIntArray(stride);
  aclDestroyIntArray(padding);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(gradInputDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
