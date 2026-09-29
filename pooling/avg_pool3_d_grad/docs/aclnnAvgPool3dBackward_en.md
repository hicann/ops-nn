# aclnnAvgPool3dBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/pooling/avg_pool3_d_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- This API performs the backpropagation of 3D average pooling and calculates the input gradient of the forward propagation of 3D average pooling.

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

  If ceilMode is true and the following condition is met:

  $$
  (D_{out} - 1) * stride[0] >= D_{in} + padding[0]
  $$
  
  Then, the shape of D_{out} needs to be reduced by 1. The same rule applies to H_{out},W_{out}.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnAvgPool3dBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnAvgPool3dBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnAvgPool3dBackwardGetWorkspaceSize(
  const aclTensor   *gradOutput,
  const aclTensor   *self,
  const aclIntArray *kernelSize,
  const aclIntArray *stride,
  const aclIntArray *padding,
  bool               ceilMode,
  bool               countIncludePad,
  int64_t            divisorOverride,
  aclTensor         *output,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnAvgPool3dBackward(
  void               *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor      *executor,
  const aclrtStream  stream)
```

## aclnnAvgPool3dBackwardGetWorkspaceSize

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
      <td>-</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Input in the forward process, corresponding to input in the formula.</td>
      <td>-</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>Input</td>
      <td>Size of the pooling window, corresponding to k in the formula.</td>
      <td>The length is 1 (KD = KH = KW) or 3 (KD, KH, KW). The value must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>Input</td>
      <td>Step size of the pooling operation, which is strides in the formula.</td>
      <td>The length is 0 (the value is the same as that of kernelSize), 1 (SD = SH = SW), or 3 (SD, SH, SW). The value must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Input</td>
      <td>Number of zero-padded layers in the D, H, and W directions of the input, which is paddings in the formula.</td>
      <td>The length is 1 (PD = PH = PW) or 3 (PD, PH, PW). The value range is [0, kernelSize/2].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceilMode</td>
      <td>Input</td>
      <td>Whether the derived output shape is rounded up.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>countIncludePad</td>
      <td>Input</td>
      <td>Whether to include the padding 0s when calculating average pooling.</td>
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
      <td>This attribute does not take effect when the value is 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>The shape is the same as that of the input self.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>4-5</td>
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
      <td>The input gradOutput, self, kernelSize, padding, or output is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type or dimension of the input gradOutput, self, or output is not supported.</td>
    </tr>
    <tr>
      <td>The dimensions of the input kernelSize, stride, and padding are not within the supported range.</td>
    </tr>
    <tr>
      <td>The value of a dimension of the input kernelSize or stride is less than or equal to 0, or the value of a dimension of the padding is less than 0.</td>
    </tr>
    <tr>
      <td>The padding attribute exceeds 1/2 of the corresponding position of the kernelSize. For example, paddingH = 2, kernelSizeH = 2, and paddingH>kernelSizeH*1/2.</td>
    </tr>
    <tr>
      <td>The shape of the output is inconsistent with that of self.</td>
    </tr>
    <tr>
      <td>The shape of gradOutput calculated based on the average pooling semantics is inconsistent with that of gradOutput passed by the API.</td>
    </tr>
  </tbody>
  </table>

## aclnnAvgPool3dBackward

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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnAvgPool3dBackwardGetWorkspaceSize.</td>
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
  - **aclnnAvgPool3dBackward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
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
