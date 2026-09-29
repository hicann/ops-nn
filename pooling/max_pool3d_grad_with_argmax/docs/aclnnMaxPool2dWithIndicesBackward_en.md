# aclnnMaxPool2dWithIndicesBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/pooling/max_pool3d_grad_with_argmax)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

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

  - If the start position of the upper left corner of the sliding window is in the padding area below or on the right, or is out of the image (no valid value can be obtained), the sliding window result will be discarded. In this case, you need to subtract 1 from the shape of the corresponding spatial axis based on the preceding formula.
    $$
    \begin{cases}
    H_{out}=H_{out} - 1& \text{if } (H_{out}-1)*s_h>=H_{in}+padding\_size_{Htop} \\
    W_{out}=W_{out} - 1& \text{if } (W_{out}-1)*s_w>=W_{in}+padding\_size_{Wleft}  \\
    \end{cases}\\
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnMaxPool2dWithIndicesBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize(
  const aclTensor   *gradOutput,
  const aclTensor   *self,
  const aclTensor   *indices,
  const aclIntArray *kernelSize, 
  const aclIntArray *stride,
  const aclIntArray *padding,
  const aclIntArray *dilation,
  bool               ceilMode,
  aclTensor         *gradInput,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnMaxPool2dWithIndicesBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize

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
    <td>gradOutput</td>
    <td>Input</td>
    <td>Gradient output from the previous step during backpropagation.</td>
    <td>The shape is the same as that of the forward output, and the data format is the same as that of self.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND, NCHW, NHWC.</td>
    <td>3-4</td>
    <td>√</td>
  </tr>
  <tr>
    <td>self</td>
    <td>Input</td>
    <td>Forward input data.</td>
    <td>-</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND, NCHW, NHWC</td>
    <td>3-4</td>
    <td>√</td>
  </tr>
  <tr>
    <td>indices</td>
    <td>Input</td>
    <td>Index of the forward output.</td>
    <td>The shape is the same as that of gradOutput. The index is the position of the maximum element in the forward output. The data format is the same as that of self.</td>
    <td>INT32 or INT64</td>
    <td>ND, NCHW, NHWC</td>
    <td>3-4</td>
    <td>√</td>
  </tr>
  <tr>
    <td>kernelSize</td>
    <td>Input</td>
    <td>Size of the sliding window used in the pooling operation.</td>
    <td>The length can only be 1 or 2.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stride</td>
    <td>Input</td>
    <td>Step for moving the window.</td>
    <td>The length can only be 0, 1, or 2. When the stride length is **0**, the value of **stride** is equal to that of **kernelSize**.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>Input</td>
    <td>Padding of the input data, indicating the number of padded elements in each dimension. It affects the behavior of the pooling window to cover the entire input tensor.</td>
    <td>The length can only be 1 or 2.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>dilation</td>
    <td>Input</td>
    <td>Dilation factor of the pooling operation. The dilation operation increases the distance between elements in the pooling window.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ceilMode</td>
    <td>Input</td>
    <td>Rounding method used to calculate the output shape.</td>
    <td>If the value is True, the output shape is rounded up. If the value is False, the output shape is rounded down.</td>
    <td>BOOL</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gradInput</td>
    <td>Output</td>
    <td>Gradient output by backpropagation.</td>
    <td>The shape and data format are the same as those of self.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND, NCHW, NHWC.</td>
    <td>3-4</td>
    <td>√</td>
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
  Ascend 950PR/Ascend 950DT: The data types of gradOutput and self can be FLOAT, FLOAT16, or BFLOAT16, and the data formats can be ND, NCHW, or NHWC.
                                   The data type of indices can be INT32 or INT64.
                                   The element value in dilation can be greater than 0, and 1D or 2D input is supported.

  <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data types of gradOutput and self support FLOAT. The data format can be ND or NCHW. When the input is 3D, the value of CHW is used. The data type of indices supports INT32. Only dilation (1, 1) is supported.

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
      <td>The input self and indices are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>The data types of gradOutput, self, indices, and gradInput are not supported.</td>
    </tr>
    <tr>
      <td>The data formats of gradOutput, self, indices, and gradInput are not supported.</td>
    </tr>
    <tr>
      <td>The shapes of gradOutput and indices are inconsistent, or the shapes of self and gradInput are inconsistent.</td>
    </tr>
    <tr>
      <td>The length of kernelSize is not 1 or 2.</td>
    </tr>
    <tr>
      <td>The value of kernelSize contains a value less than or equal to 0.</td>
    </tr>
    <tr>
      <td>The length of stride is not 0, 1, or 2.</td>
    </tr>
    <tr>
      <td>The value of stride contains a value less than or equal to 0.</td>
    </tr>
    <tr>
      <td>The number of padding elements is not 1 or 2</td>.
    </tr>
    <tr>
      <td>The value of padding is less than 0 or greater than the value of kernelSize.</td>
    </tr>
    <tr>
      <td>The element value of dilation does not meet the input parameter requirements.</td>
    </tr>
  </tbody>
  </table>

## aclnnMaxPool2dWithIndicesBackward

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
      <td>Size of the workspace allocated on the device, which is obtained by the first function aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize.</td>
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
  - **aclnnMaxPool2dWithIndicesBackward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

- The input data does not support NaN and -Inf. The value of **indices** cannot exceed $H\_in*W\_in$ in the formula and must be greater than or equal to 0.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
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
