# aclnnMaxPool2dWithMask

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/pooling/max_pool3d_with_argmax_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    √    |
| <term>Atlas training series products</term>                             |    √    |

## Function

- Description:
Provides two-dimensional max pooling for the input channels of the input signal, and outputs the pooled value out and indices (calculated using the mask semantics).

- Formula:

  - Calculation formula of each element of the output tensor:

    $$
    out(N_j, C_j, h, w) = \max\limits_{{m\in[0,k_{H}-1],n\in[0,k_{W}-1]}}input(N_i,C_j,stride[0]\times h + m, stride[1]\times w + n)
    $$

  - For the shape inference of out tensor:

    $$
    [N, C, H_{out}, W_{out}]=[N,C,\lfloor{\frac{H_{in}+2 \times {padding[0] - dilation[0] \times(kernelSize[0] - 1) - 1}}{stride[0]}}\rfloor + 1,\lfloor{\frac{W_{in}+2 \times {padding[1] - dilation[1] \times(kernelSize[1] - 1) - 1}}{stride[1]}}\rfloor + 1]
    $$

  - For the shape inference of indices tensor:

    $$
    [N, C, H_{indices}, W_{indices}]=[N,C,k_h \times k_w,   (\lceil{\frac{H_{out} \times W_{out}}{16}}\rceil+1) \times 2 \times 16]
    $$

## Prototype
Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnMaxPool2dWithMaskGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnMaxPool2dWithMask** is called to perform computation.

- `aclnnStatus aclnnMaxPool2dWithMaskGetWorkspaceSize(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnMaxPool2dWithMask(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnMaxPool2dWithMaskGetWorkspaceSize

- **Parameters:**
  - **self** (aclTensor*, compute input): input tensor, which is input in the formula. It is an aclTensor on the device. The shape can only be three-dimensional or four-dimensional. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. When the [data format](../../../docs/en/context/data_formats.md) is three-dimensional, ND is supported. When the data format is four-dimensional, NCHW is supported.
    - <term>Atlas inference series products</term>: The data type can be FLOAT.
    - <term>Atlas training series products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, and BFLOAT16.
  - **kernelSize** (aclIntArray*, compute input): size of the maximum pooling window, which is k in the formula. It is an aclIntArray on the host. The data type can be INT64. The array length must be 1 or 2, and all array elements must be greater than 0.
  - **stride** (aclIntArray*, compute input): stride of the window, which is **stride** in the formula. It is an aclIntArray on the host. The data type can be INT64. When the stride length is **0**, the value of **stride** is equal to that of **kernelSize**.
  - **padding** (aclIntArray*, compute input): the number of layers of padding to be applied on each side, padding_size in the formula, aclIntArray on the host, with negative infinity values used for padding. The data type can be INT64. The array length must be 1 or 2, and the array elements must be greater than or equal to 0 or less than or equal to kernelSize divided by 2.
  - **dilation** (aclIntArray*, compute input): controls the stride of elements in the window, dilation_size in the formula, aclIntArray on the host. The data type can be INT64. The value can only be 1.
  - **ceilMode** (bool, compute input): controls the value mode when the out shape is inferred. It is of the Bool type on the host. Only true or false is supported. If the value is true, round up to infer the shapes of Hout and Wout. If the value is false, round down.
  - **out** (aclTensor*, compute output): output tensor, out in the formula, aclTensor on the device. It is the pooled result. The shape needs to be calculated according to the shape inference formula of out in the function description. The [data format](../../../docs/en/context/data_formats.md) supports ND in three dimensions and NCHW in four dimensions, which is the same as that of **self**.
    - <term>Atlas inference series products</term>: The data type can be FLOAT.
    - <term>Atlas training series products</term>: The data type can be FLOAT or FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT, FLOAT16, and BFLOAT16.
  - **indices** (aclTensor*, compute output): output tensor, aclTensor on the device. This tensor consists of the index positions of maximum values (using mask semantics). The data type can only be INT8. The shape needs to be calculated according to the shape inference formula of **indices** in the function description. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported. The [data format](../../../docs/en/context/data_formats.md) supports ND in three dimensions and NCHW in four dimensions, which is the same as that of **self**. It is a user-defined mask value.
  - **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.
  - **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, out, kernelSize, stride, padding, or dilation is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self is not supported.
                                      2. The data format of self is not supported.
                                      3. The shape of self is not three-dimensional or four-dimensional.
                                      4. The data type and data format of self are inconsistent with those of out.
                                      5. The shape of self has an axis of 0 in C, H, or W.
                                      6. An axis of the shape of out deduced from the formula is less than 0.
                                      7. The shape of out inferred from the formula is inconsistent with the actual shape of out.
                                      8. The shape of indices inferred from the formula is inconsistent with the actual shape of indices.
                                      9. The length of kernelSize is not 1 or 2.
                                      10. kernelSize has values less than or equal to 0.
                                      11. The length of stride is not 0, 1, or 2.
                                      12. stride has values less than or equal to 0.
                                      13. The length of padding is not 1 or 2.
                                      14. padding has values less than 0 or greater than kernelSize divided by 2.
                                      15. The length of dilation is not 1 or 2.
                                      16. The value of dilation is not 1.
```

## aclnnMaxPool2dWithMask

- **Parameters:**
  * **workspace** (void*, input): address of the workspace to be allocated on the device.
  * **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnMaxPool2dWithMaskGetWorkspaceSize**.
  * **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.
  * **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic compute:
  - **aclnnMaxPool2dWithMask** defaults to a deterministic implementation.

- The input data does not support NaN and -Inf.

- <term>Atlas training series products</term>: When the input data is FLOAT, the data is converted to FLOAT16 for computation. As a result, the accuracy drops to some extent.

- <term>Atlas inference series products</term>: When **ceilMode** is set to **True**, the following stride scenarios are not supported:

$$s_h >= (H_{in} + padding\_size) / (H_{out} - 1)$$

$$s_w >= (W_{in} + padding\_size) / (W_{out} - 1)$$

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_pool2d_with_indices.h"

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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Use CHECK as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> selfShape = {1, 1, 4, 3};
  std::vector<int64_t> outShape = {1, 1, 2, 1};
  std::vector<int64_t> indicesShape = {1, 1, 4, 64};
  std::vector<int64_t> kernelSizeData = {2, 2};
  std::vector<int64_t> strideData = {2, 2};
  std::vector<int64_t> paddingData = {0, 0};
  std::vector<int64_t> dilationData = {1, 1};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* indices = nullptr;
  std::vector<float> selfHostData = {0.0850, -0.5147, -0.0212, -0.5654, -0.3222, 0.5847, 1.7510, 0.9954, 0.1842, 0.8392, 0.4835, 0.9213};
  std::vector<float> outHostData = {0, 0};
  std::vector<int8_t> indicesHostData(256, 0);

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT8, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an input array.
  aclIntArray* kernelSize = aclCreateIntArray(kernelSizeData.data(), 2);
  aclIntArray* stride = aclCreateIntArray(strideData.data(), 2);
  aclIntArray* padding = aclCreateIntArray(paddingData.data(), 2);
  aclIntArray* dilation = aclCreateIntArray(dilationData.data(), 2);
  const bool ceilMode = false;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnMaxPool2dWithMask API call example
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  // Call the first-phase API of aclnnMaxPool2dWithMask.
  ret = aclnnMaxPool2dWithMaskGetWorkspaceSize(self, kernelSize, stride, padding, dilation, ceilMode, out, indices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPool2dWithMaskGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMaxPool2dWithMask.
  ret = aclnnMaxPool2dWithMask(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPool2dWithMask failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  size = GetShapeSize(indicesShape);
  std::vector<int8_t> indicesResultData(size, 0);
  ret = aclrtMemcpy(indicesResultData.data(), indicesResultData.size() * sizeof(indicesResultData[0]), indicesDeviceAddr,
                    size * sizeof(indicesResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy indices result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, indicesResultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(indices);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
