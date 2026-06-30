# aclnnAvgPool3dBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/pooling/avg_pool3_d_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    ×     |

## Function

- Description: Performs backpropagation of 3D average pooling, that is, to compute the input gradient of forward propagation of 3D average pooling.

- Formula:
  The relationship among the input($N,C,D_{in},H_{in},W_{in}$), gradOutput($N,C,D_{out},H_{out},W_{out}$), pooling stride ($stride$), and kernelSize ($kD,kH,kW$) during backpropagation is as follows:

  $$
  D_{in} = (D_{out} - 1) * {stride[0]} + kernel\_size[0] - 2 * padding[0]
  $$

  $$
  H_{in} = (H_{out} - 1) * {stride[1]} + kernel\_size[1] - 2 * padding[1]
  $$

  $$
  W_{in} = (W_{out} - 1) * {stride[2]} + kernel\_size[2] - 2 * padding[2]
  $$

  If **ceil_mode** is **true** and the following condition is met:

  $$
  (D_{out} - 1) * stride[0] >= D_{in} + padding[0]
  $$

  Then, the shape of D_{out} needs to be reduced by 1. The same rule applies to H_{out},W_{out}.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnAvgPool3dBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnAvgPool3dBackward** is called to perform computation.

- `aclnnStatus aclnnAvgPool3dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding, bool ceilMode, bool countIncludePad, int64_t divisorOverride, aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnAvgPool3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnAvgPool3dBackwardGetWorkspaceSize

- **Parameters:**
  - **gradOutput** (aclTensor*, compute input): input gradient, which is an aclTensor on the NPU device. The gradient can be 4D ($C,D_{out},H_{out},W_{out}$) or 5D ($N,C,D_{out},H_{out},W_{out}$). N indicates the batch size, C indicates the number of channels, and D, H, and W indicate the depth, height, and width, respectively. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. The data type can be BFLOAT16, FLOAT16, or FLOAT32, and must be the same as that of **self**.
  - **self** (aclTensor*, compute input): input data during the forward process ($input$ in the formula), which is an aclTensor on the NPU device. The shape can be 4D ($C,D_{in},H_{in},W_{in}$) or 5D ($N,C,D_{in},H_{in},W_{in}$). [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. The data type can be BFLOAT16, FLOAT16, or FLOAT32, and must be the same as that of **gradOutput**.
  - **kernelSize** (aclIntArray*, compute input): aclIntArray on the host, indicating the pooling window size. The value is an INT64 array, with a length of 1 (KD = KH = KW) or 3 (KD, KH, KW). The value must be greater than 0.
  - **stride** (aclIntArray*, compute input): aclIntArray on the host, indicating the stride of the pooling operation. The value is an INT64 array, with a length of 0 (the value is the same as that of kernelSize), 1 (SD = SH = SW), or 3 (SD, SH, SW). The value must be greater than 0.
  - **padding** (aclIntArray*, compute input): aclIntArray on the host, indicating the number of zero-padding layers in the D, H, and W directions of the input. The value is an INT64 array, with a length of 1 (PD = PH = PW) or 3 (PD, PH, PW). The value is within the range of [0, kernelSize/2].
  - **ceilMode** (bool, compute input): whether to round up the shape of the output deduced during forward average pooling. (The value **True** indicates to round up.) The data type can be BOOL.
  - **countIncludePad** (bool, compute input): whether to include padded zeros when computing forward average pooling. (The value **True** indicates to include padded zeros.) The data type can be BOOL.
  - **divisorOverride** (int64_t, compute input): divisor for computing average pooling. If specified, it is used as the divisor in the average pooling. When the value is **0**, this attribute does not take effect. The data type can be INT64.
  - **output** (aclTensor *, compute output): aclTensor on the device. The shape is the same as that of **self**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND. The data type and [data format](../../../docs/en/context/data_formats.md) must be the same as those of **gradOutput**. The data type can be BFLOAT16, FLOAT16, or FLOAT32.
  - **workspaceSize** (uint64_t \*, output): size of the workspace to be allocated on the device.
  - **executor** (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed gradOutput, self, kernelSize, padding, or output is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or shape of gradOutput, self, or output is not supported.
                                    2. The shape of kernelSize, stride, or padding is not supported.
                                    3. The value of a dimension of the passed kernelSize or stride is less than or equal to 0, or the value of a dimension of padding is less than 0.
                                    4. The padding attribute exceeds 1/2 of the position corresponding to kernelSize. For example, paddingH = 2, kernelSizeH = 2, paddingH > kernelSizeH × 1/2.
                                    5. The shape of output is different from that of self.
                                    6. The gradOutput shape computed based on average pooling is inconsistent with that passed by the API.
  ```

## aclnnAvgPool3dBackward

- **Parameters:**

  - **workspace** (void\*, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API **aclnnAvgPool3dBackwardGetWorkspaceSize**.
  - **executor** (aclOpExecutor\*, input): operator executor, containing the operator computation process.
  - **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic compute:
  - **aclnnAvgPool3dBackward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

## Example
The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_avgpool3d_backward.h"

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
  std::vector<int64_t> gradOutputShape = {1, 16, 1, 1, 1};
  std::vector<int64_t> selfShape = {1, 16, 4, 4, 4};
  std::vector<int64_t> kernelDims = {4, 4, 4};
  std::vector<int64_t> strideDims = {1, 1, 1};
  std::vector<int64_t> paddingDims = {0, 0, 0};
  std::vector<int64_t> outputShape = {1, 16, 4, 4, 4};
  bool ceilMode = false;
  int64_t divisorOverride = 0;
  bool countIncludePad = false;

  void* gradOutputDeviceAddr = nullptr;
  void* selfAddr = nullptr;
  void* outputAddr = nullptr;

  aclTensor* gradOutput = nullptr;
  aclTensor* selfInput = nullptr;
  aclTensor* output = nullptr;

  std::vector<float> gradOutputHostData(GetShapeSize(gradOutputShape), 1);
  std::vector<float> selfHostData(GetShapeSize(selfShape), 1);
  std::vector<float> outputHostData(GetShapeSize(outputShape), 1);
  // Create a gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an inputshape aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfAddr, aclDataType::ACL_FLOAT,
  &selfInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an output.
  ret = CreateAclTensor(outputHostData, outputShape, &outputAddr, aclDataType::ACL_FLOAT, &output);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a kernel aclIntArray.
  aclIntArray *kernelSize = aclCreateIntArray(kernelDims.data(), 3);

  // Create a stride aclIntArray.
  aclIntArray *stride = aclCreateIntArray(strideDims.data(), 3);

  // Create a paddings aclIntArray.
  aclIntArray *padding = aclCreateIntArray(paddingDims.data(), 3);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAvgPool3dBackward.
  ret = aclnnAvgPool3dBackwardGetWorkspaceSize(gradOutput,
                                        selfInput,
                                        kernelSize,
                                        stride,
                                        padding,
                                        ceilMode,
                                        countIncludePad,
                                        divisorOverride,
                                        output,
                                        &workspaceSize,
                                        &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool3dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAvgPool3dBackward.
  ret = aclnnAvgPool3dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool3dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outputShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outputAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, outData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(selfInput);
  aclDestroyTensor(output);
  aclDestroyIntArray(kernelSize);
  aclDestroyIntArray(stride);
  aclDestroyIntArray(padding);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfAddr);
  aclrtFree(outputAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
