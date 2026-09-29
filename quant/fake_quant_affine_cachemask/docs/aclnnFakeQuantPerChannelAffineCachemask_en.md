# aclnnFakeQuantPerChannelAffineCachemask

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Function: Fake-quantizes the input data self on the specified axis using scale and zero_point, updates the value range of the fake-quantized output based on quant_min and quant_max, and returns the result out and the corresponding position mask.
- Formula: Computes the temporary variable **qval**, then calculates **out** and **mask**.

  - The data type of zero_point is int32.
  $$
  qval = Round(std::nearby\_int(self / scale) + zero\_point)
  $$
  - The data type of zero_point is float16 or float32.
  $$
  qval = std::lrintf(self / scale + zero\_point)       
  $$

  $$
  out = (Min(quant\_max, Max(quant\_min, qval)) - zero\_point) * scale
  $$

  $$
  mask = (qval >= quant\_min)   \&  (qval <= quant\_max)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnFakeQuantPerChannelAffineCachemask** is called to perform computation.

```Cpp
aclnnStatus aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize(
  const aclTensor   *self,
  const aclTensor   *scale,
  const aclTensor   *zeroPoint,
  int64_t            axis,
  int64_t            quantMin,
  int64_t            quantMax,
  aclTensor         *out,
  aclTensor         *mask,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnFakeQuantPerChannelAffineCachemask(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize

- **Parameters**

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
        <td>`self` in the formula.</td>
        <td>-</td>
        <td>FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>scale</td>
        <td>Input</td>
        <td>Scale in the formula, indicating the scaling coefficient of the fake-quantized input.</td>
        <td>The size must be the same as that of zeroPoint and the size of self on the axis.</td>
        <td>FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>zeroPoint</td>
        <td>Input</td>
        <td>Zero point in the formula, indicating the zero reference parameter of the fake-quantized input.</td>
        <td>The size must be the same as that of scale and the size of self on the axis.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>axis</td>
        <td>Input</td>
        <td>Indicates the computation dimension.</td>
        <td>The value range is [–self.dim(), self.dim() – 1].</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMin</td>
        <td>Input</td>
        <td>Indicates the minimum value of the fake-quantized input data.</td>
        <td>The value must be less than or equal to quantMax.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMax</td>
        <td>Input</td>
        <td>Indicates the maximum value of the fake-quantized input data.</td>
        <td>The value must be greater than or equal to quantMin.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>`out` in the formula.</td>
        <td>The shape must be the same as that of `self`.</td>
        <td>FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>mask</td>
        <td>Output</td>
        <td>Mask in the formula.</td>
        <td>The shape must be the same as that of `self`.</td>
        <td>BOOL</td>
        <td>ND</td>
        <td>0-8</td>
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
- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

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
      <td>The input self, scale, zeroPoint, out, or mask is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type of self, scale, zeroPoint, out, or mask is not supported.</td>
    </tr>
    <tr>
      <td>The shape of out or mask is inconsistent with that of self.</td>
    </tr>
    <tr>
      <td>The shape of scale or zeroPoint is not 1D.</td>
    </tr>
    <tr>
      <td>The sizes of scale and zeroPoint are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimension specified by axis is not within the valid range [–self.dim(), self.dim() – 1].</td>
    </tr>
    <tr>
      <td>The size of scale or zeroPoint is inconsistent with that of self on the axis.</td>
    </tr>
    <tr>
      <td>quantMin is greater than quantMax.</td>
    </tr>
  </tbody>
  </table>

## aclnnFakeQuantPerChannelAffineCachemask

- **Parameters**
  
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
        <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize API.</td>
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

- Deterministic computation:
  - **aclnnFakeQuantPerChannelAffineCachemask** defaults to a deterministic implementation.

- The current operator **FakeQuantPerChannelAffineCachemask** does not yet support FLOAT32 or FLOAT16 inputs for **zero_point**. This constraint is currently enforced at the aclnn API level and will be lifted once the operator natively supports these data types.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
  // 2. Construct the inputs and outputs based on the API definition.
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

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
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
  
  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify the code based on the API definition.
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
