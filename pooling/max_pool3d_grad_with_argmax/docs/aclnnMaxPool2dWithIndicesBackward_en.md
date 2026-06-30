# aclnnMaxPool2dWithIndicesBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/pooling/max_pool3d_grad_with_argmax)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    ×    |
| <term>Atlas training series products</term>                             |    ×    |

## Function

- Description:
Performs backpropagation of the [aclnnMaxPool2dWithIndices](../../max_pool3d_with_argmax_v2/docs/aclnnMaxPool2dWithIndices_en.md) operator.
- Deduction formula of the input tensor:
  - When **ceilMode** is set to **False**, the H and W dimensions in the shape of the indices tensor are derived as follows:

    $$
    [H_{out}, W_{out}]=[\lfloor{\frac{H_{in}+  padding\_size_{Htop} + padding\_size_{Hbottom} - {dilation\_size \times(k_h - 1) - 1}}{s_h}}\rfloor + 1,\lfloor{\frac{W_{in}+ padding\_size_{Wleft} + padding\_size_{Wright} - {dilation\_size \times(k_w - 1) - 1}}{s_w}}\rfloor + 1]
    $$

  - When **ceilMode** is set to **True**, the H and W dimensions in the shape of the out tensor are derived as follows:

    $$
    [H_{out}, W_{out}]=[\lceil{\frac{H_{in}+  padding\_size_{Htop} + padding\_size_{Hbottom} - {dilation\_size \times(k_h - 1) - 1}}{s_h}}\rceil + 1,\lceil{\frac{W_{in}+ padding\_size_{Wleft} + padding\_size_{Wright} - {dilation\_size \times(k_w - 1) - 1}}{s_w}}\rceil + 1]
    $$

  - If the upper left corner of the sliding window starts from the lower or right padding or goes off-bounds (no valid value can be obtained), the sliding window result is discarded. The shape of the corresponding spatial axis needs to be subtracted by 1 based on the preceding derivation formula.

    $$
    \begin{cases}
    H_{out}=H_{out} - 1& \text{if } (H_{out}-1)*s_h>=H_{in}+padding\_size_{Htop} \\
    W_{out}=W_{out} - 1& \text{if } (W_{out}-1)*s_w>=W_{in}+padding\_size_{Wleft}  \\
    \end{cases}\\
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnMaxPool2dWithIndicesBackward** is called to perform computation.

- `aclnnStatus aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, const aclIntArray *kernelSize, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnMaxPool2dWithIndicesBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize

- **Parameters:**
  * **gradOutput** (const aclTensor \*, compute input): output gradient in the previous step during backpropagation, aclTensor on the device. The shape is the same as that of the forward output. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The data format is the same as that of self.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT. The [data format](../../../docs/en/context/data_formats.md) can be NCHW or CHW.
  * **self**(const aclTensor \*, compute input): input data of forward propagation, aclTensor on the device. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can be FLOAT. The [data format](../../../docs/en/context/data_formats.md) can be NCHW or CHW.
  * **indices** (aclTensor \*, compute input): forward output index, aclTensor on the device. The shape is the same as that of the input **gradOutput**. It indicates the index of the maximum element in the forward output. The data format is the same as that of **self**.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can only be INT32. The [data format](../../../docs/en/context/data_formats.md) can be NCHW or CHW.
  * **kernelSize** (const aclIntArray \*, compute input): size of the sliding window used in the pooling operation, aclIntArray on the host. The length can only be 1 or 2.
    - When the number of elements in **kernelSize** is **1**, the window size is (kernelSize[0], kernelSize[0]).
    - When the number of elements in **kernelSize** is **2**, the window size is (kernelSize[0], kernelSize[1]).
  * **stride** (aclIntArray*, compute input): stride of the window, aclIntArray on the host. The length can only be 0, 1, or 2. When the stride length is **0**, the value of **stride** is equal to that of **kernelSize**.
    - When the number of elements in **stride** is 0, the stride length is the same as **kernelSize**.
    - When the number of elements in **stride** is **1**, the stride length is (stride[0], stride[0]).
    - When the number of elements in **stride** is **2**, the stride length is (stride[0], stride[1]).
  * **padding** (const aclIntArray \*, compute input): padding of the input data, indicating the padding amount in each dimension of the input. It affects how the pooling window covers the entire input tensor. It is an aclIntArray on the host. The length can only be 1 or 2.
    - When the number of elements in **padding** is **1**, the head and tail of the H and W axes are padded with `-Inf`, with the padding length equal to padding[0].
    - When the number of elements in **padding** is **2**, the head and tail of the H axis are padded with `-Inf` of length padding[0], and the head and tail of the W axis are padded with `-Inf` of length padding[1].
  * **dilation** (const aclIntArray \*, compute input): dilation factor of the pooling operation. The dilation operation increases the distance between elements in the pooling window. It is an aclIntArray on the host.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: Only dilation (1, 1) is supported.
  * **ceilMode** (const bool \*, compute input): controls whether the output size of the pooling operation is rounded up. It is a boolean parameter on the host. If the value is **True**, the output shape is calculated by rounding up. If the value is **False**, the output shape is calculated by rounding down.
  * **gradInput** (aclTensor \*, compute output): gradient of the output for backpropagation, aclTensor on the device. The shape is the same as that of **self**. The data format is the same as that of **self**.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The data type can only be FLOAT. The [data format](../../../docs/en/context/data_formats.md) can be NCHW or CHW.
  * **workspaceSize** (uint64_t \*, output): size of the workspace to be allocated on the device.
  * **executor** (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self or indices is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of gradOutput, self, indices, or gradInput is not supported.
                                   2. The data format of gradOutput, self, indices, or gradInput is not supported.
                                   3. The shape of gradOutput is different from that of indices, and the shape of self is different from that of gradInput.
                                   4. The length of kernelSize is not 1 or 2.
                                   5. kernelSize has values less than or equal to 0.
                                   6. The length of stride is not 0, 1, or 2.
                                   8. stride has values less than or equal to 0.
                                   9. The number of elements in padding is not 1 or 2.
                                   10. padding has values less than 0 or greater than kernelSize divided by 2.
                                   11. The element values in dilation do not meet the input parameter requirements.
```

## aclnnMaxPool2dWithIndicesBackward

- **Parameters:**
  * **workspace** (void \*, input): address of the workspace to be allocated on the device.
  * **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize**.
  * **executor** (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  * **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic compute:
  - **aclnnMaxPool2dWithIndicesBackward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

- The input data does not support NaN and -Inf. The value of **indices** cannot exceed $H\_in*W\_in$ in the formula and must be greater than or equal to 0.

## Example
The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_pool2d_with_indices_backward.h"

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
  std::vector<int64_t> gradOutShape = {1, 1, 2, 1};
  std::vector<int64_t> selfShape = {1, 1, 4, 3};
  std::vector<int64_t> indicesShape = {1, 1, 2, 1};
  std::vector<int64_t> gradInShape = {1, 1, 4, 3};
  std::vector<int64_t> kernelSizeData = {2, 2};
  std::vector<int64_t> strideData = {2, 2};
  std::vector<int64_t> paddingData = {0, 0};
  std::vector<int64_t> dilationData = {1, 1};
  void* gradOutDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* gradInDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* self = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* gradIn = nullptr;
  std::vector<float> gradOutHostData = {0.4757, 0.1726};
  std::vector<float> selfHostData = {0.0850, -0.5147, -0.0212, -0.5654, -0.3222, 0.5847, 1.7510, 0.9954, 0.1842, 0.8392, 0.4835, 0.9213};
  std::vector<int32_t> indicesHostData = {0, 6};
  std::vector<float> gradInHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create a gradOut aclTensor.
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a gradIn aclTensor.
  ret = CreateAclTensor(gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT, &gradIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an input array.
  aclIntArray* kernelSize = aclCreateIntArray(kernelSizeData.data(), 2);
  aclIntArray* stride = aclCreateIntArray(strideData.data(), 2);
  aclIntArray* padding = aclCreateIntArray(paddingData.data(), 2);
  aclIntArray* dilation = aclCreateIntArray(dilationData.data(), 2);
  const bool ceilMode = false;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnMaxPool2dWithIndicesBackward API call example
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  // Call the first-phase API of aclnnMaxPool2dWithIndicesBackward.
  ret = aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize(gradOut, self, indices, kernelSize, stride, padding, dilation, ceilMode, gradIn, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMaxPool2dWithIndicesBackward.
  ret = aclnnMaxPool2dWithIndicesBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPool2dWithIndicesBackward failed. ERROR: %d\n", ret); return ret);

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
