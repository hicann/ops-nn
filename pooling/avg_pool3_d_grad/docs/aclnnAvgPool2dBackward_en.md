# aclnnAvgPool2dBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:58:00.269Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Backward propagation of 2D average pooling. Computes the gradient of the input for the forward propagation of 2D average pooling.

- Formula: Assume that the input tensor of the 2D average pooling forward pass is $X$, the output tensor is $Y$, the kernel size is $k*k$, and the stride is $s$. Then the gradient of $X$, $\frac{\partial L}{\partial X}$, is calculated as follows:

  $$
  \frac{\partial L}{\partial X_{i,j}}=\frac{1}{k^2}\sum_{n=0}^{k-1}\frac{\partial L}{\partial Y_{\lfloor\frac{i*s+m}{k}\rfloor,\lfloor\frac{j*s+n}{k}\rfloor}}
  $$

  The parameters are described as follows:

  * $L$ is the loss function, and $\lfloor\cdot\rfloor$ indicates rounding up.
  * $X_{i,j}$ represents the $i$-th row and $j$-th column of the input feature map.
  * $Y_{\lfloor\frac{i*s+m}{k}\rfloor,\lfloor\frac{j*s+n}{k}\rfloor}$ represents the pixel value at the $\lfloor\frac{i*s+m}{k}\rfloor$-th row and $\lfloor\frac{j*s+n}{k}\rfloor$-th column of the output feature map.
  * $k$ represents the kernel size of the pooling window.
  * $s$ represents the stride size.
  * $\frac{\partial L}{\partial X_{i,j}}$ represents the partial derivative of the loss function $L$ with respect to the pixel value at row $i$, column $j$ in the input feature map.
  * $\frac{\partial L}{\partial Y_{\lfloor\frac{i*s+m}{k}\rfloor,\lfloor\frac{j*s+n}{k}\rfloor}}$ represents the partial derivative of the loss function $L$ with respect to the pixel value at row $\lfloor\frac{i*s+m}{k}\rfloor$, column $\lfloor\frac{j*s+n}{k}\rfloor$ in the output feature map.

## Function Prototype

Each operator is split into a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnAvgPool2dBackwardGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the "aclnnAvgPool2dBackward" API to perform the computation.

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

- **Parameters**:
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
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
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
      <td>Input in the forward propagation.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW, NCL</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>Input</td>
      <td>Pooling window size, which is k in the formula.</td>
      <td>The length is 1 (kH=kW) or 2 (kH, kW), and the value is greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>Input</td>
      <td>Stride of the pooling operation, i.e., strides in the formula.</td>
      <td>The length is 0 (stride=kernelSize), 1 (sH=sW), or 2 (sH, sW). The value must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Input</td>
      <td>Number of zero-padding layers on the H and W dimensions of the input, corresponding to paddings in the formula.</td>
      <td>Length is 1 (padH=padW) or 2 (padH, padW), with values in the range [0, kernelSize/2].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceilMode</td>
      <td>Input</td>
      <td>Whether the shape of the deduced output out is rounded up.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>countIncludePad</td>
      <td>Input</td>
      <td>Whether the zeros filled by padding are included in average pooling computation.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divisorOverride</td>
      <td>Input</td>
      <td>Divisor used to compute the average.</td>
      <!-- npu="910,310p" id8 -->
      <td><term>Atlas inference products</term>, <term>Atlas training products</term>: The value range is [-255,255]. 0 indicates that it does not affect whether padding participates in the average computation.</td>
      <!-- end id8 -->
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Specifies the computation logic of the Cube unit.</td>
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
      <td>Returns the workspace size that needs to be applied by the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>
  <!-- npu="910,310p" id9 -->
  <term>Atlas inference products</term>, <term>Atlas training products</term>: The data types of the self and out parameters do not support BFLOAT16.
  <!-- end id9 -->

  cubeMathType: The supported enumeration values are as follows:
    * 0: KEEP_DTYPE, which keeps the input data type for computation.
      <!-- npu="910,310p" id10 -->
      * <term>Atlas training products</term>, <term>Atlas inference products</term>: This option is not supported when the input data type is FLOAT32.
      <!-- end id10 -->
    * 1: ALLOW_FP32_DOWN_PRECISION, which allows the input data to be computed at reduced precision.
      <!-- npu="910,310p" id11 -->
      * <term>Atlas training products</term>, <term>Atlas inference products</term>: When the input data type is FLOAT32, it is converted to FLOAT16 for computation. No processing is performed for other input data types.
      <!-- end id11 -->
      <!-- npu="A3,910b" id12 -->
      * <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. No processing is performed for other input data types.
      <!-- end id12 -->
      <!-- npu="950" id13 -->
      * <term>Ascend 950PR&950DT products</term>: In globalPooling mode, when the input data type is FLOAT32, it is converted to HFLOAT32 for computation. No processing is performed for other input data types.
      <!-- end id13 -->
    * 2: USE_FP16, supports downscaling the input precision to FLOAT16 for computation.
      <!-- npu="A3,910b" id14 -->
      * <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: This option is not supported when the input data type is BFLOAT16.
      <!-- end id14 -->
      <!-- npu="950" id15 -->
      * <term>Ascend 950PR&950DT products</term>: In globalPooling mode, this option is not supported when the input data type is BFLOAT16.
      <!-- end id15 -->
    * 3: USE_HF32, supports reducing input precision to HFLOAT32 for computation.
      <!-- npu="910,310p" id16 -->
      * <term>Atlas training products</term>, <term>Atlas inference products</term>: This option is not supported.
      <!-- end id16 -->
      <!-- npu="A3,910b" id17 -->
      * <term>Atlas A2 products</term>, <term>Atlas A3 products</term>   : When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. This option is not supported for other input data types.
      <!-- end id17 -->
      <!-- npu="950" id18 -->
      * <term>Ascend 950PR&950DT products</term>: In globalPooling mode, when the input data type is FLOAT32, it is converted to HFLOAT32 for computation. This option is not supported for other input data types.
      <!-- end id18 -->
- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed-in gradOutput, self, kernelSize, padding, or gradInput is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type/data format of the passed-in gradOutput or gradInput is out of the supported range.</td>
    </tr>
    <tr>
      <td>The data type or data format of the passed-in self and gradInput are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimensions of the passed-in gradOutput, self, kernelSize, stride, or gradInput are out of the supported range.</td>
    </tr>
    <tr>
      <td>A dimension value of the passed-in gradOutput, self, kernelSize, stride, or gradInput is less than 0.</td>
    </tr>
    <tr>
      <td>The passed-in cubeMathType is out of the supported range.</td>
    </tr>
    <tr>
      <td>The padding attribute exceeds 1/2 of the value at the corresponding position in kernelSize. For example, paddingH=2 and kernelSizeH=2, where paddingH>kernelSizeH*1/2.</td>
    </tr>
    <tr>
      <td>The passed in divisorOverride is out of the supported range, but this check is not performed for Ascend 950PR&950DT products.</td>
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
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address applied by the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied by the Device side, obtained by the first-phase API aclnnAvgPool2dBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the operator, containing the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  <!-- npu="A3,910b,310p,910" id19 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>: aclnnAvgPool2dBackward defaults to a non-deterministic implementation. Determinism can be enabled via aclrtCtxSetSysParamOpt.
  <!-- end id19 -->
  <!-- npu="950" id20 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnAvgPool2dBackward defaults to a deterministic implementation.
  <!-- end id20 -->

<!-- npu="910,310p" id7 -->
- <term>Atlas training products</term>, <term>Atlas inference products</term>: The Cube unit does not support FLOAT32 computation. When the input is FLOAT32, set cubeMathType=1 (ALLOW_FP32_DOWN_PRECISION) to allow the API to internally cast to FLOAT16 for computation.

<!-- end id7 -->

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  // Boilerplate for resource initialization.
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
  // Call aclrtMemcpy to copy data from the host to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of a contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, Format, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the list of ACL external APIs.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output as required by the API.
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

  // Create the gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput, aclFormat::ACL_FORMAT_NCHW);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self, aclFormat::ACL_FORMAT_NCHW);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the gradInput aclTensor.
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput, aclFormat::ACL_FORMAT_NCHW);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the kernel aclIntArray.
  aclIntArray *kernelSize = aclCreateIntArray(kernelDims.data(), 2);

  // Create the stride aclIntArray.
  aclIntArray *stride = aclCreateIntArray(strideDims.data(), 2);

  // Create the paddings aclIntArray.
  aclIntArray *padding = aclCreateIntArray(paddingDims.data(), 2);

  // 3. Call the CANN operator library API. Replace the name with the specific API name.
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
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnAvgPool2dBackward.
  ret = aclnnAvgPool2dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool2dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait synchronously for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the result from the device memory to the host memory. Modify the code based on the specific API definition.
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), gradInputDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, outData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(gradInput);
  aclDestroyIntArray(kernelSize);
  aclDestroyIntArray(stride);
  aclDestroyIntArray(padding);

  // 7. Release device resources. Modify the code based on the specific API definition.
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
