# aclnnMaxPool3dWithArgmaxBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:02:14.115Z -->

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
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

Backward propagation of the forward max pooling [aclnnMaxPool3dWithArgmax](../../max_pool3d_with_argmax_v2/docs/aclnnMaxPool3dWithArgmax.md). Gradients are backfilled to the coordinates of the maximum value in each window, and accumulated at the same coordinates.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnMaxPool3dWithArgmaxBackwardGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the operator computation process. Then call the "aclnnMaxPool3dWithArgmaxBackward" API to execute the computation.

```Cpp
aclnnStatus aclnnMaxPool3dWithArgmaxBackwardGetWorkspaceSize(
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
aclnnStatus aclnnMaxPool3dWithArgmaxBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnMaxPool3dWithArgmaxBackwardGetWorkspaceSize

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
    <td>The gradient output from the previous step in backward propagation.</td>
    <td>Same shape as the forward output, and same data format as self.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>4-5</td>
    <td>√</td>
  </tr>
  <tr>
    <td>self</td>
    <td>Input</td>
    <td>The forward input data.</td>
    <td>-</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>4-5</td>
    <td>√</td>
  </tr>
  <tr>
    <td>indices</td>
    <td>Input</td>
    <td>Indices of the maximum elements in the forward input.</td>
    <td>-</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    <td>4-5</td>
    <td>√</td>
  </tr>
  <tr>
    <td>kernelSize</td>
    <td>Input</td>
    <td>Size of the sliding window used in the pooling operation.</td>
    <td>1 (kD = kH = kW) or 3 (kD, kH, kW)</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stride</td>
    <td>Input</td>
    <td>Stride at which the window moves.</td>
    <td>Length: 0 (sD = kD, sH = kH, sW = kW), 1 (sD = sH = sW), or 3 (sD, sH, sW).</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>Input</td>
    <td>Number of zero-padding layers along the D, H, and W dimensions of the input.</td>
    <td>Length 1 (padD = padH = padW) or 3 (padD, padH, padW).</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>dilation</td>
    <td>Input</td>
    <td>Stride of elements within the window.</td>
    <td>The length is 1 or 3, and only 1 is supported.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ceilMode</td>
    <td>Input</td>
    <td>Rounding method used when computing the output shape.</td>
    <td>If True, the output shape is rounded up. If False, the output shape is rounded down.</td>
    <td>BOOL</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gradInput</td>
    <td>Output</td>
    <td>Gradient output by backward propagation.</td>
    <td>The shape and data format are the same as those of self.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>4-5</td>
    <td>√</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Returns the size of the workspace to be allocated on the Device.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Returns the op executor that contains the operator computation flow.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  </tbody></table>

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: The data type of `indices` does not support INT64. depth * height * width does not support values greater than max int32.

  <!-- end id7 -->

- **Return value**

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
      <td>The input self or indices is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>The data type of gradOutput, self, indices, or gradInput is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data format of gradOutput, self, indices, or gradInput is not in the supported range.</td>
    </tr>
    <tr>
      <td>The shape of gradOutput is inconsistent with that of indices, or the shape of self is inconsistent with that of gradInput.</td>
    </tr>
    <tr>
      <td>The length of kernelSize is not equal to 1 or 3.</td>
    </tr>
    <tr>
      <td>kernelSize contains a value that is less than or equal to 0.</td>
    </tr>
    <tr>
      <td>The length of stride is not equal to 0, 1, or 3.</td>
    </tr>
    <tr>
      <td>stride contains a value that is less than or equal to 0.</td>
    </tr>
    <tr>
      <td>The length of padding is not equal to 1 or 3.</td>
    </tr>
    <tr>
      <td>padding contains a value that is less than 0 or greater than kernelSize/2.</td>
    </tr>
    <tr>
      <td>The length of dilation is not equal to 1 or 3.</td>
    </tr>
    <tr>
      <td>The platform does not support this.</td>
    </tr>
  </tbody>
  </table>

## aclnnMaxPool3dWithArgmaxBackward

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnMaxPool3dWithArgmaxBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: aclnnMaxPool3dWithArgmaxBackward defaults to a non-deterministic implementation and supports enabling deterministic computation through aclrtCtxSetSysParamOpt.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnMaxPool3dWithArgmaxBackward defaults to a deterministic implementation.
  <!-- end id12 -->

- Unsupported type description:
  - DOUBLE: The instruction does not support DOUBLE.
  - Whether empty tensors are supported: empty input yields empty output is not supported.

- Boundary value scenario description:
  - When the input is Inf, the output is Inf.
  - When the input is NaN, the output is NaN.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_pool3d_with_argmax_backward.h"

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
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API reference.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. The construction must be customized according to the API.
  std::vector<int64_t> gradOutShape = {1, 1, 1, 1, 1};
  std::vector<int64_t> selfShape = {1, 1, 2, 2, 2};
  std::vector<int64_t> indicesShape = {1, 1, 1, 1, 1};
  std::vector<int64_t> gradInShape = {1, 1, 2, 2, 2};
  std::vector<int64_t> kernelSizeData = {2, 2, 2};
  std::vector<int64_t> strideData = {2, 2, 2};
  std::vector<int64_t> paddingData = {0, 0, 0};
  std::vector<int64_t> dilationData = {1, 1, 1};
  void* gradOutDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* gradInDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* self = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* gradIn = nullptr;
  std::vector<float> gradOutHostData = {0.4757};
  std::vector<float> selfHostData = {0.0850, -0.5147, -0.0212, -0.5654, -0.3222, 0.5847, 1.7510, 0.9954};
  std::vector<int32_t> indicesHostData = {6};
  std::vector<float> gradInHostData = {0, 0, 0, 0, 0, 0, 0, 0};

  // Create the gradOut aclTensor.
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the gradIn aclTensor.
  ret = CreateAclTensor(gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT, &gradIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the input arrays.
  aclIntArray* kernelSize = aclCreateIntArray(kernelSizeData.data(), 3);
  aclIntArray* stride = aclCreateIntArray(strideData.data(), 3);
  aclIntArray* padding = aclCreateIntArray(paddingData.data(), 3);
  aclIntArray* dilation = aclCreateIntArray(dilationData.data(), 3);
  const bool ceilMode = false;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnMaxPool3dWithArgmaxBackward API call example
  // 3. Call the CANN operator library API. Replace the API name with the specific one.
  // Call the aclnnMaxPool3dWithArgmaxBackwardGetWorkspaceSize first-phase API.
  ret = aclnnMaxPool3dWithArgmaxBackwardGetWorkspaceSize(gradOut, self, indices, kernelSize, stride, padding, dilation, ceilMode, gradIn, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPool3dWithArgmaxBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnMaxPool3dWithArgmaxBackward second-phase API.
  ret = aclnnMaxPool3dWithArgmaxBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPool3dWithArgmaxBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result in device memory to the host. Modify the code based on the specific API definition.
  auto size = GetShapeSize(gradInShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradIn result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(gradOut);
  aclDestroyTensor(self);
  aclDestroyTensor(indices);
  aclDestroyTensor(gradIn);

  // 7. Release device resources. Modify the code based on the specific API definition.
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
