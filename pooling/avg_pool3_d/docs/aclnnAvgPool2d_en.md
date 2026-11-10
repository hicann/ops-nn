# aclnnAvgPool2d

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:53:55.973Z -->

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

- Description: Performs a 2D average pooling operation on the input tensor with a window of $kH * kW$ and a stride of $sH * sW$, where $k$ is kernelSize, representing the pooling window size, and $s$ is stride, representing the step size of the pooling operation.
- Formula:
  The relationship between the input ($N,C,H,W$), output ($N,C,H_{out},W_{out}$), pooling stride ($strides$), and pooling window size ($kH,kW$) is:

  $$
  H_{out}=\lfloor \frac{H_{in}+2*paddings[0]-kH}{strides[0]}+1 \rfloor
  $$

  $$
  W_{out}=\lfloor \frac{W_{in}+2*paddings[1]-kW}{strides[1]}+1 \rfloor
  $$

  If ceilMode is true and the following condition is met:

  $$
  (H_{out} - 1) * stride[0] >= H_{in} + padding[0]
  $$

  then the shape of H_{out} needs to be reduced by 1. The same applies to W_{out}.

  $$
  out(N_i,C_i,h,w)=\frac{1}{kH*kW}\sum_{m=0}^{kH-1}\sum_{n=0}^{kW-1}input(N_i,C_i,strides[0]*h+m,strides[1]*w+n)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnAvgPool2dGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnAvgPool2d" API to perform the computation.

```Cpp
aclnnStatus aclnnAvgPool2dGetWorkspaceSize(
  const aclTensor   *self,
  const aclIntArray *kernelSize,
  const aclIntArray *strides,
  const aclIntArray *paddings,
  const bool         ceilMode,
  const bool         countIncludePad,
  const int64_t      divisorOverride,
  const int8_t       cubeMathType,
  aclTensor         *out,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnAvgPool2d(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAvgPool2dGetWorkspaceSize

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
      <td>self</td>
      <td>Input</td>
      <td>Represents the tensor to be converted, which is the input in the formula.</td>
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
      <td>strides</td>
      <td>Input</td>
      <td>Stride of the pooling operation, that is, strides in the formula.</td>
      <td>The length is 0 (sH=kH, sW=kW), 1 (sH=sW), or 2 (sH, sW), and the value must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>Input</td>
      <td>Number of padding layers filled with 0 in the H and W directions of the input, which is paddings in the formula.</td>
      <td>The length is 1 (padH=padW) or 2 (padH, padW), with values in the range [0, kernelSize/2].</td>
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
      <td>Whether the zeros filled by padding are included when computing average pooling.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divisorOverride</td>
      <td>Input</td>
      <td>The divisor used to compute the average.</td>
      <td>If set to 0, the feature is disabled.</td>
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
      <td>out</td>
      <td>Output</td>
      <td>The output tensor, which is out in the formula.</td>
      <td>-</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW, NCL</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be applied by users on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>
  <!-- npu="910,310p" id12 -->
  <term>Atlas inference products</term>, <term>Atlas training products</term>: The data types of the self and out parameters do not support BFLOAT16.
  <!-- end id12 -->

  cubeMathType: The supported enumeration values are as follows:
    * 0: KEEP_DTYPE, which keeps the input data type for computation.
      <!-- npu="910,310p" id8 -->
      * <term>Atlas training products</term>, <term>Atlas inference products</term>: This option is not supported when the input data type is FLOAT32.
      <!-- end id8 -->
    * 1: ALLOW_FP32_DOWN_PRECISION, which allows the input data to be computed at reduced precision.
      <!-- npu="910,310p" id9 -->
      * <term>Atlas training products</term>, <term>Atlas inference products</term>: When the input data type is FLOAT32, it is converted to FLOAT16 for computation. No processing is performed for other input data types.
      <!-- end id9 -->
      <!-- npu="A3,910b" id10 -->
      * <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: This option is not supported when the input data type is FLOAT32. No processing is performed for other input data types.
      <!-- end id10 -->
    * 2: USE_FP16, supports downscaling the input precision to FLOAT16 for computation.
      <!-- npu="A3,910b" id11 -->
      * <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: This option is not supported when the input data type is BFLOAT16.
      <!-- end id11 -->
    * 3: USE_HF32. This option is not supported.
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
      <th>description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The self, kernelSize, strides, paddings, or out passed in is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type or data format of self or out passed in is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data types or data formats of self and out passed in are inconsistent.</td>
    </tr>
    <tr>
      <td>The value of a dimension in self, paddings, or out passed in is less than 0.</td>
    </tr>
    <tr>
      <td>The value of a dimension in kernelSize or strides passed in is less than or equal to 0.</td>
    </tr>
    <tr>
      <td>The length of passed-in kernelSize or paddings is less than 1.</td>
    </tr>
    <tr>
      <td>The output shape calculated based on average pooling semantics is inconsistent with the specified shape.</td>
    </tr>
    <tr>
      <td>The attribute padding exceeds 1/2 of the corresponding position of kernelSize, for example, paddingH=2, kernelSizeH=2, paddingH>kernelSizeH*1/2.</td>
    </tr>
  </tbody>
  </table>

## aclnnAvgPool2d

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
      <td>Workspace memory address applied by the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied by the Device, obtained by calling the first-phase API aclnnAvgPool2dGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream on which tasks are executed.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnAvgPool2d defaults to a deterministic implementation.

<!-- npu="910" id7 -->
- <term>Atlas training products</term>: The Cube unit does not support FLOAT32 computation. When the input is FLOAT32, set cubeMathType=1 (ALLOW_FP32_DOWN_PRECISION) to allow the API to internally cast to FLOAT16 for computation.

<!-- end id7 -->

## Example

The following sample code is for reference only. For details about the compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_avgpool2d.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                     \
      if (!(cond)) {                       \
          Finalize(deviceId, stream);      \
          return_expr;                     \
      }                                    \
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
  // Boilerplate: resource initialization.
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
  // Call aclrtMalloc to apply for memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the memory on the device side.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnAvgPool2dTest(int32_t deviceId, aclrtStream& stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output, which need to be custom constructed according to the API.
  int64_t divisorOverride = 0;
  bool countIncludePad = true;
  bool ceilMode = false;
  int8_t cubeMathType = 2;

  std::vector<int64_t> selfShape = {1, 16, 4, 4};
  std::vector<int64_t> outShape = {1, 16, 1, 1};

  std::vector<int64_t> kernelDims = {4, 4};
  std::vector<int64_t> strideDims = {1, 1};
  std::vector<int64_t> paddingDims = {0, 0};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData(256, 2);
  std::vector<float> outHostData(16, 0);

  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> selfTensorPtr(self, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create the kernel aclIntArray.
  aclIntArray *kernelSize = aclCreateIntArray(kernelDims.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> kernelSizePtr(kernelSize, aclDestroyIntArray);
  CHECK_FREE_RET(kernelSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // Create the stride aclIntArray.
  aclIntArray *strides = aclCreateIntArray(strideDims.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(strides, aclDestroyIntArray);
  CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // Create the paddings aclIntArray.
  aclIntArray *paddings = aclCreateIntArray(paddingDims.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> paddingsPtr(paddings, aclDestroyIntArray);
  CHECK_FREE_RET(paddings != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call a CANN operator library API, and change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAvgPool2d.
  ret = aclnnAvgPool2dGetWorkspaceSize(self,
                                       kernelSize,
                                       strides,
                                       paddings,
                                       ceilMode,
                                       countIncludePad,
                                       divisorOverride,
                                       cubeMathType,
                                       out,
                                       &workspaceSize,
                                       &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool2dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnAvgPool2d.
  ret = aclnnAvgPool2d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool2d failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, and copy the result from device memory to host memory. Modify the code according to the definition of the specific API.
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, outData[i]);
  }
  return ACL_SUCCESS;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnAvgPool2dTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool2dTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
