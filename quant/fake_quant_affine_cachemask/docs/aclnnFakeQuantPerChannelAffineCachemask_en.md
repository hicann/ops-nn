# aclnnFakeQuantPerChannelAffineCachemask

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/quant/fake_quant_affine_cachemask)

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    ×     |

## Function

- Description: Performs fake quantization on the input **self** along the specified axis with **scale** and **zero_point**, limits the value range of the fake-quantized output by **quant_min** and **quant_max**, and finally returns the output **out** as well as the corresponding position mask **mask**.
- Formula: Computes the temporary variable **qval**, then calculates **out** and **mask**.

  $$
  qval = Round(std::nearby\_int(self / scale) + zero\_point)
  $$

  $$
  out = (Min(quant\_max, Max(quant\_min, qval)) - zero\_point) * scale
  $$

  $$
  mask = (qval >= quant\_min)   \&  (qval <= quant\_max)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnFakeQuantPerChannelAffineCachemask** is called to perform computation.

- `aclnnStatus aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize(const aclTensor* self, const aclTensor* scale, const aclTensor* zeroPoint, int64_t axis, int64_t quantMin, int64_t quantMax, aclTensor* out, aclTensor* mask, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnFakeQuantPerChannelAffineCachemask(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize

- **Parameters**
  
  - **self** (aclTensor\*, input): input tensor, corresponding to `self` in the formula. This tensor is a device-side aclTensor. The data type can be FLOAT16 or FLOAT32. It supports [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md). Its [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **scale** (aclTensor\*, input): scaling coefficient for input fake quantization, corresponding to `scale` in the formula. This tensor is a device-side aclTensor. The data type can be FLOAT16 or FLOAT32, and the shape must be one-dimensional. The size must be consistent with that of **zeroPoint** and with the size of **self** along the axis dimension. It supports [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md). Its [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **zeroPoint** (aclTensor\*, input): zero-point parameter for input fake quantization, corresponding to `zero_point` in the formula. This tensor is a device-side aclTensor. The data type can be INT32, and the shape must be one-dimensional. The size must be consistent with that of **scale** and with the size of **self** along the axis dimension. It supports [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md). Its [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **axis** (int64_t, input): host-side integer that specifies the computation dimension. The value range is [–self.dim(), self.dim() – 1].
  - **quantMin** (int64_t, input): host-side integer that represents the minimum value after fake quantization of the input data. It must be less than or equal to **quantMax**.
  - **quantMax** (int64_t, input): host-side integer that represents the maximum value after fake quantization of the input data. It must be greater than or equal to **quantMin**.
  - **out** (aclTensor\*, output): output tensor, corresponding to `out` in the formula. This tensor is a device-side aclTensor. The data type can be FLOAT16 or FLOAT32, and the shape must be the same as that of `self`. It supports [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md). Its [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **mask** (aclTensor\*, output): mask tensor, corresponding to `mask` in the formula. This tensor is a device-side aclTensor. The data type must be BOOL, and the shape must be the same as that of `self`. It supports [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md). Its [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **workspaceSize** (uint64_t\*, output): size of the workspace to be allocated on the device.
  - **executor** (aclOpExecutor\*\*, output): operator executor, containing the operator computation flow.
- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  
  ```
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self, scale, zeroPoint, out, or mask is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self, scale, zeroPoint, out, or mask is not supported.
                                        2. The shape of out or mask does not match that of self.
                                        3. The shape of scale or zeroPoint is not one-dimensional.
                                        4. The sizes of scale and zeroPoint do not match.
                                        5. The dimension specified by axis is not within the valid range [–self.dim(), self.dim() – 1].
                                        6. The size of scale or zeroPoint does not match that of self on axis.
                                        7. quantMin is greater than quantMax.
  ```

## aclnnFakeQuantPerChannelAffineCachemask

- **Parameters**
  
  - **workspace** (void\*, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained via the first-phase API **aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize**.
  - **executor** (aclOpExecutor\*, input): operator executor, containing the operator computation flow.
  - **stream** (aclrtStream, input): stream for executing the task.
- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic computation:
  - **aclnnFakeQuantPerChannelAffineCachemask** defaults to a deterministic implementation.

- The current operator **FakeQuantPerChannelAffineCachemask** does not yet support FLOAT32 or FLOAT16 inputs for **zero_point**. This constraint is currently enforced at the aclnn API level and will be lifted once the operator natively supports these data types.


## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fake_quant_per_channel_affine_cachemask.h"

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
  // (Boilerplate) Initialize resources.
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

  // Calculate the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID (deviceId) based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
  // 2. Construct inputs and outputs based on API definitions.
  std::vector<int64_t> selfShape = {1};
  std::vector<int64_t> scaleShape = {1};
  std::vector<int64_t> zeroPointShape = {1};
  std::vector<int64_t> outShape = {1};
  std::vector<int64_t> maskShape = {1};
  void* selfDeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  void* zeroPointDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* maskDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* scale = nullptr;
  aclTensor* zeroPoint = nullptr;
  aclTensor* out = nullptr;
  aclTensor* mask = nullptr;
  std::vector<float> selfHostData{1};
  std::vector<float> scaleHostData{1};
  std::vector<int32_t> zeroPointHostData{1};
  std::vector<float> outHostData{1};
  std::vector<char> maskHostData{1};
  int64_t quantMin = 1;
  int64_t quantMax = 3;
  int axis = 0;
  // Create an aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(zeroPointHostData, zeroPointShape, &zeroPointDeviceAddr, aclDataType::ACL_INT32, &zeroPoint);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(maskHostData, maskShape, &maskDeviceAddr, aclDataType::ACL_BOOL, &mask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Modify the API as required.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnEye.
  ret = aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize(self, scale, zeroPoint, axis, quantMin, quantMax, out, mask, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnFakeQuantPerChannelAffineCachemask.
  ret = aclnnFakeQuantPerChannelAffineCachemask(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFakeQuantPerChannelAffineCachemask failed. ERROR: %d\n", ret); return ret);
  
  // 4. (Boilerplate) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(scale);
  aclDestroyTensor(zeroPoint);
  aclDestroyTensor(out);
  aclDestroyTensor(mask);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(scaleDeviceAddr);
  aclrtFree(zeroPointDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(maskDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
