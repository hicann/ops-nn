# aclnnAvgPool2dBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/pooling/avg_pool3_d_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Function: performs the backpropagation of 2D average pooling and calculates the input gradient of the forward propagation of 2D average pooling.

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

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAvgPool2dBackwardGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnAvgPool2dBackward` is called to perform computation.

```Cpp
aclnnStatus aclnnAvgPool2dBackwardGetWorkspaceSize(
  const aclTensor   *gradOutput,
  const aclTensor   *self,
  const aclIntArray *kernelSize,
  const aclIntArray *stride,
  const aclIntArray *padding,
  bool               ceilMode,
  bool               countIncludePad,
  int64_t            divisorOverride,
  int8_t             cubeMathType,
  aclTensor         *gradInput,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnAvgPool2dBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAvgPool2dBackwardGetWorkspaceSize

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
      <td>Input gradient.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW, NCL</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Input in the forward process.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW, NCL</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>Input</td>
      <td>Size of the pooling window, which is k in the formula.</td>
      <td>The value is 1 (kH = kW) or 2 (kH, kW), and must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>Input</td>
      <td>Stride of the pooling operation, which is strides in the formula.</td>
      <td>The value is 1 (sH = sW) or 2 (sH, sW), and must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Input</td>
      <td>Number of zero-padded layers in the height and width directions of the input, which is paddings in the formula.</td>
      <td>The value is 1 (padH = padW) or 2 (padH, padW), and cannot be less than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceilMode</td>
      <td>Input</td>
      <td>Whether the shape of the derived output out is rounded up.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>countIncludePad</td>
      <td>Input</td>
      <td>Whether to include the 0s in the padding when calculating average pooling.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divisorOverride</td>
      <td>Input</td>
      <td>Divisor for calculating the average.</td>
      <td>For <term>Atlas inference products</term> and <term>Atlas training products</term>: The value range is [–255, 255]. The value 0 indicates that the padding is not involved in the average calculation.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Specifies the compute logic of the Cube unit.</td>
      <td>-</td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>Output</td>
      <td>Output tensor, which is out in the formula.</td>
      <td>-</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW, NCL</td>
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
  For <term>Atlas inference products</term> and <term>Atlas training products</term>, the data types of the self and out parameters do not support BFLOAT16.

  cubeMathType: The following enumerated values are supported:
    * 0: KEEP_DTYPE. The input data type is retained for computation.
      * <term>Atlas training products</term> and <term>Atlas inference products</term>: This option is not supported when the input data type is FLOAT32.
    * 1: ALLOW_FP32_DOWN_PRECISION. The input data can be computed with reduced precision.
      * <term>Atlas training products</term> and <term>Atlas inference products</term>: If the input data type is FLOAT32, it will be converted to FLOAT16 for computation. When the input is of other data types, it is not processed.
      * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: If the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, it is not processed.
      * Ascend 950PR/Ascend 950DT: In global pooling mode, when the input data type is FLOAT32, the data is converted to HFLOAT32 for computation. When the input is of other data types, it is not processed.
    * 2: USE_FP16. The input data can be downgraded to FLOAT16 for computation.
      * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: This option is not supported when the input data type is BFLOAT16.
      * Ascend 950PR/Ascend 950DT: In global pooling mode, this option is not supported when the input data type is BFLOAT16.
    * 3: USE_HF32. The input data can be downgraded to HFLOAT32 for computation.
      * <term>Atlas training products</term> and <term>Atlas inference products</term>: This option is not supported.
      * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: If the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, this option is not supported.
      * Ascend 950PR/Ascend 950DT: In global pooling mode, when the input data type is FLOAT32, it will be converted to HFLOAT32 for computation. When the input is of other data types, this option is not supported.
- **Returns:**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.
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
      <td>The input gradOutput, self, kernelSize, padding, or gradInput is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type or format of the input gradOutput or gradInput is not supported.</td>
    </tr>
    <tr>
      <td>The data type or format of the input self is inconsistent with that of the input gradInput.</td>
    </tr>
    <tr>
      <td>The dimensions of the input gradOutput, self, kernelSize, stride, or gradInput are not supported.</td>
    </tr>
    <tr>
      <td>The value of a dimension of the input gradOutput, self, kernelSize, stride, or gradInput is less than 0.</td>
    </tr>
    <tr>
      <td>The input cubeMathType is not supported.</td>
    </tr>
    <tr>
      <td>The padding attribute exceeds half of the kernel size. For example, paddingH = 2, kernelSizeH = 2, and paddingH>kernelSizeH*1/2.</td>
    </tr>
    <tr>
      <td>The input divisorOverride is not supported, but this verification is not performed on Ascend 950PR/Ascend 950DT.</td>
    </tr>
  </tbody>
  </table>

## aclnnAvgPool2dBackward

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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnAvgPool2dBackwardGetWorkspaceSize.</td>
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

- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - **aclnnAvgPool2dBackward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

- <term>Atlas training products</term> and <term>Atlas inference products</term>: The Cube unit does not support FLOAT32 computation. The input data type FLOAT32 can be converted to FLOAT16 in the API for computation by setting **cubeMathType** to **1** (**ALLOW_FP32_DOWN_PRECISION**).

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
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
