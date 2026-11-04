# aclnnFakeQuantPerTensorAffineCachemask

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:27:42.679Z -->

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

- Description:
  - fake_quant_enabled >= 1: For the input data self, performs fake quantization on the input self using scale and zero_point, clamps the fake quantization output to the range defined by quant_min and quant_max, and finally returns the result out and the corresponding position mask.
  - fake_quant_enabled < 1: Returns out as a self.clone() object, with the mask being all True.
- Formula: When fake_quant_enabled >= 1, a temporary variable qval is first computed based on the operator functionality, and then out and mask are derived from it.

  $$
  qval = Round(std::nearby\_int(self / scale) + zero\_point)
  $$

  $$
  out = (Min(quant\_max, Max(quant\_min, qval)) - zero\_point) * scale
  $$

  $$
  mask = (qval >= quant\_min)   \&  (qval <= quant\_max)
  $$

- Special-value semantics (aligned with PyTorch `fake_quantize_per_channel_affine_cachemask`):
  - NaN: out = (quant\_min - zero\_point) * scale, mask = False
  - +Inf: out = (quant\_max - zero\_point) * scale, mask = False
  - -Inf: out = (quant\_min - zero\_point) * scale, mask = False

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnFakeQuantPerTensorAffineCachemask" API first to obtain the input parameters and calculate the required workspace size following the procedure, and then call the "aclnnFakeQuantPerTensorAffineCachemask" API to execute the computation.

```Cpp
aclnnStatus aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize(
  const aclTensor   *self,
  const aclTensor   *scale,
  const aclTensor   *zeroPoint,
  float              fakeQuantEnabled,
  int64_t            quantMin,
  int64_t            quantMax,
  aclTensor         *out,
  aclTensor         *mask,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnFakeQuantPerTensorAffineCachemask(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
    <col style="width: 149px">
    <col style="width: 121px">
    <col style="width: 304px">
    <col style="width: 253px">
    <col style="width: 222px">
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
       <td>self in the formula.</td>
       <td>-</td>
       <td>FLOAT16, FLOAT32</td>
       <td>ND</td>
       <td>0-8</td>
       <td>√</td>
      </tr>
      <tr>
       <td>scale</td>
       <td>Input</td>
       <td>scale in the formula, indicates the scaling factor of the input pseudo quantization.</td>
       <td>The size is 1.</td>
       <td>FLOAT16, FLOAT32</td>
       <td>ND</td>
       <td>1</td>
       <td>√</td>
      </tr>
      <tr>
       <td>zeroPoint</td>
       <td>Input</td>
       <td>zero_point in the formula, indicates the input zero point parameter for fake quantization.</td>
       <td>The size is 1.</td>
       <td>INT32</td>
       <td>ND</td>
       <td>1</td>
       <td>√</td>
      </tr>
      <tr>
       <td>fakeQuantEnabled</td>
       <td>Input</td>
       <td>Indicates whether to perform fake quantization computation.</td>
       <td>-</td>
       <td>FLOAT32</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>quantMin</td>
       <td>Input</td>
       <td>Indicates the minimum value of the input data after pseudo-quantization.</td>
       <td>Must be less than or equal to quantMax.</td>
       <td>INT64</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>quantMax</td>
       <td>Input</td>
       <td>Indicates the maximum value of the input data after pseudo-quantization.</td>
       <td>Must be greater than or equal to quantMin.</td>
       <td>INT64</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>out</td>
       <td>Output</td>
       <td>The out in the formula.</td>
       <td>The shape must be the same as that of `self`.</td>
       <td>FLOAT16, FLOAT32</td>
       <td>ND</td>
       <td>0-8</td>
       <td>√</td>
      </tr>
      <tr>
       <td>mask</td>
       <td>Output</td>
       <td>mask in the formula.</td>
       <td>The shape must be the same as that of `self`.</td>
       <td>BOOL</td>
       <td>ND</td>
       <td>0-8</td>
       <td>√</td>
      </tr>
      <tr>
       <td>workspaceSize</td>
       <td>Output</td>
       <td>Returns the workspace size that needs to be applied by on the Device side.</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>executor</td>
       <td>Output</td>
       <td>Returns the op executor, including the operator computation flow.</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
    </tbody></table>

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
      <td>self, scale, zeroPoint, out, or mask is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of self, scale, zeroPoint, out, or mask are not within the supported range.</td>
    </tr>
    <tr>
      <td>The size of scale or zeroPoint is not 1.</td>
    </tr>
    <tr>
      <td>The shapes of out and mask are inconsistent with that of self.</td>
    </tr>
    <tr>
      <td>quantMin is greater than quantMax.</td>
    </tr>
  </tbody>
  </table>

## aclnnFakeQuantPerTensorAffineCachemask

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
        <td>Workspace memory address applied on the Device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace applied on the Device. The size is obtained by the first-phase API aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for task execution.</td>
      </tr>
    </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnFakeQuantPerTensorAffineCachemask defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fake_quant_per_tensor_affine_cachemask.h"

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
  // Call aclrtMalloc to apply for device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Build them as required by the API.
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
  float fakeQuantEnabled = 1.0f;
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

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize.
  ret = aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize(self, scale, zeroPoint, fakeQuantEnabled, quantMin, quantMax, out, mask, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnFakeQuantPerTensorAffineCachemask.
  ret = aclnnFakeQuantPerTensorAffineCachemask(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFakeQuantPerTensorAffineCachemask failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from device memory to the host, and modify the code based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and modify the code based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(scale);
  aclDestroyTensor(zeroPoint);
  aclDestroyTensor(out);
  aclDestroyTensor(mask);

  // 7. Release device resources and modify the code based on the specific API definition.
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
