# aclnnMaxPool2dWithMask

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/pooling/max_pool3d_with_argmax_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    √    |
| <term>Atlas training products</term>                             |    √    |

## Function

- API function:
For the input channel of the input signal, the 2D max pooling operation is provided to output the pooled value out and index indices (calculated based on the mask semantics, which refers to the mask information that records the position of the maximum value during pooling).

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

```Cpp
aclnnStatus aclnnMaxPool2dWithMaskGetWorkspaceSize(
  const aclTensor   *self,
  const aclIntArray *kernelSize, 
  const aclIntArray *stride,
  const aclIntArray *padding,
  const aclIntArray *dilation,
  bool               ceilMode,
  aclTensor         *out,
  aclTensor         *indices,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnMaxPool2dWithMask(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnMaxPool2dWithMaskGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 264px">
  <col style="width: 253px">
  <col style="width: 262px">
  <col style="width: 148px">
  <col style="width: 135px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Tensor to be computed.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>NCHW and ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>Input</td>
      <td>Size of the max pooling window.</td>
      <td>The length is 1 or 2, and all elements in the array must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>Input</td>
      <td>Step for moving the window.</td>
      <td>When the stride length is 0, the stride value is equal to the value of kernelSize.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Input</td>
      <td>Number of layers added to each edge. The position to be added is filled with "negative infinity".</td>
      <td>The array length must be 1 or 2, and all elements in the array must be greater than or equal to 0 or less than or equal to kernelSize/2.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>Input</td>
      <td>Stride of the elements in the window.</td>
      <td>The value can only be 1.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceilMode</td>
      <td>Input</td>
      <td>Rounding method used for calculating the output shape.</td>
      <td>If the value is True, the output shape is rounded up. If the value is False, the output shape is rounded down.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor, which is the result after pooling.</td>
      <td>The shape is derived from the preceding formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>NCHW and ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Output</td>
      <td>Tensor consisting of the index positions of the maximum values (using the mask semantics).</td>
      <td>The shape is derived from the preceding formula, and is a user-defined mask value.</td>
      <td>INT8.</td>
      <td>NCHW and ND</td>
      <td>3-4</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - <term>Atlas inference products</term>: The data type can be FLOAT.
  - <term>Atlas training products</term>: The data type cannot be BFLOAT16.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input self, out, kernelSize, stride, padding, or dilation is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="16">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="16">161002</td>
      <td>The data type of self is not supported.</td>
    </tr>
    <tr>
      <td>The data format of self is not supported.</td>
    </tr>
    <tr>
      <td>The shape of self is not 3D or 4D.</td>
    </tr>
    <tr>
      <td>The data types and formats of self and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The shape of self has an axis of 0 in C, H, or W.</td>
    </tr>
    <tr>
      <td>The length of an axis in the shape of out derived from the formula is less than 0.</td>
    </tr>
    <tr>
      <td>The shape of out derived from the formula is inconsistent with the actual shape of out.</td>
    </tr>
    <tr>
      <td>The shape of indices derived from the formula is inconsistent with the actual shape of indices.</td>
    </tr>
    <tr>
      <td>The length of kernelSize is not 1 or 2.</td>
    </tr>
    <tr>
      <td>The value in kernelSize is less than or equal to 0.</td>
    </tr>
    <tr>
      <td>The length of stride is not 0, 1, or 2.</td>
    </tr>
    <tr>
      <td>The value in stride is less than or equal to 0.</td>
    </tr>
    <tr>
      <td>The length of padding is not 1 or 2</td>.
    </tr>
    <tr>
      <td>The value in padding is less than 0 or greater than kernelSize/2.</td>
    </tr>
    <tr>
      <td>The length of dilation is not 1 or 2.</td>
    </tr>
    <tr>
      <td>The value in dilation is not 1.</td>
    </tr>
  </tbody>
  </table>

## aclnnMaxPool2dWithMask

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnMaxPool2dWithMaskGetWorkspaceSize API.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>
- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - **aclnnMaxPool2dWithMask** defaults to a deterministic implementation.

- The input data does not support NaN and -Inf.

- <term>Atlas training products</term>: When the input data is FLOAT, the data is converted to FLOAT16 for computation. As a result, the accuracy drops to some extent.

- <term>Atlas inference products</term>: When **ceilMode** is set to **True**, the following stride scenarios are not supported:

$$s_h >= (H_{in} + padding\_size) / (H_{out} - 1)$$

$$s_w >= (W_{in} + padding\_size) / (W_{out} - 1)$$

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
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
