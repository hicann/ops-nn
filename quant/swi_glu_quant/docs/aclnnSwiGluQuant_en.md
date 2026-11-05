# aclnnSwiGluQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:40:01.118Z -->

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

- Description: Adds the quant operation after the SwiGlu activation function to perform SwiGluQuant computation on input x.
- Supported scope: SwiGluQuant currently **supports only MoE scenarios**. The inputs x and group_index of SwiGluQuant come from the outputs of the GroupedMatMul operator and MoeInitRouting. The group_index parameter enables MoE group-based dynamic quantization, static per_tensor quantization, and static per_channel quantization.
- Dynamic quantization formula:

  $$
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp}[0\colon g[0],\colon] = Act[0\colon g[0],\colon] * smooth\_scales[0,\colon], i=0 \\
    Y_{tmp}[g[i]\colon g[i+1], \colon] = Act[g[i]\colon g[i+1], \colon] *  smooth\_scales[i+1, \colon], i \in (0, G) \cap \mathbb{Z}\\
    scale=dstTypeScale/row\_max(abs(Y_{tmp}))
  $$

  $
    Y = Cast(Mul(Y_{tmp}, Scale))
  $
     where A represents the first half of input x, B represents the second half of input x, g represents group_index, and G is the number of groups in group_index. When quantizing to int8, $dstTypeScale = 127$ (the maximum value of int8); when quantizing to int4, $dstTypeScale = 7$ (the maximum value of int4). INT4 output packs two values per byte and ranges from -8 to 7.

- Static quantization formula:

  $
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp}[0\colon g[0],\colon] = Act[0\colon g[0],\colon] * smooth\_scales[0,\colon] + offsets[0,\colon], i=0 \\
    Y_{tmp}[g[i]\colon g[i+1], \colon] = Act[g[i]\colon g[i+1], \colon] *  smooth\_scales[i+1, \colon] + offsets[i+1, \colon], i \in (0, G) \cap \mathbb{Z}\\
  $
  $
    Y = Cast(Y_{tmp})
  $
  where A represents the first half of input x, B represents the second half of input x, g represents group_index, and G is the number of groups in group_index.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnSwiGluQuantGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation process. Then call the "aclnnSwiGluQuant" API to perform the computation.

```Cpp
aclnnStatus aclnnSwiGluQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *smoothScalesOptional,
  const aclTensor *offsetsOptional,
  const aclTensor *groupIndexOptional,
  bool             activateLeft,
  char            *quantModeOptional,
  const aclTensor *yOut,
  const aclTensor *scaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSwiGluQuant(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnSwiGluQuantGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
  <col style="width: 301px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 300px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
  <col style="width: 145px">
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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input data to be processed, x in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The last dimension of x must be a multiple of 2, and the number of dimensions of x must be greater than 1. Currently, the last dimension length of input x must not exceed 8192.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>smoothScalesOptional (aclTensor*)</td>
      <td>Input</td>
      <td>smooth_scales for quantization, smooth_scales in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape supports [G, N] and [G, ], where G indicates the number of groups in groupIndex, and N is half the size of the last dimension of the computation input x. In the current version, user-provided null pointers are not supported.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
     <tr>
      <td>offsetsOptional (aclTensor*)</td>
      <td>Input</td>
      <td>offsets in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>This parameter does not take effect in dynamic quantization use cases. A user-provided null pointer is sufficient.</li><li>In static quantization use cases, the supported data type is FLOAT.</li><li>In per_channel mode, the supported shape is [G, N].</li><li>In per_tensor mode, the supported shape is [G, ], and the data type and shape must keep consistent with smoothScalesOptional.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input</td>
      <td>group_index required for MoE grouping, that is, group_index in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The supported shape is [G, ]. Elements in group_index must be non-decreasing, and the maximum value must not exceed the product of all dimension sizes of input x except the last dimension; the value of G must not exceed the product of all dimension sizes of input x except the last dimension.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
      <tr>
      <td>activateLeft (bool)</td>
      <td>Input</td>
      <td>Computation input.</td>
      <td>true indicates activating the first half of input x, and false indicates activating the second half of input x.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>quantModeOptional (char*)</td>
      <td>Input</td>
      <td>Computation input.</td>
      <td>"static" indicates static quantization, "dynamic" indicates dynamic quantization, and "dynamic_msd" indicates dynamic MSD quantization. Currently, only "dynamic" dynamic quantization and "static" static quantization are supported. Static quantization supports only per_tensor quantization and per_channel quantization. A user-provided null pointer indicates dynamic quantization.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>The size of the last dimension of output yOut is half that of the last dimension of input x, and the other dimensions keep consistent with x. When yOut is INT4, the last dimension of x must be divisible by 4.</td>
      <td>INT8, INT4</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
      <tr>
      <td>scaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>Compared with input x, the shape of output scaleOut has no last dimension, and the other dimensions keep consistent with input x.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
      <tr>
      <td>workspaceSize(uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, including the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x or yOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The input or output data type is not within the supported range.</td>
    </tr>
    <tr>
      <td>The dimension of the input or output parameter is not within the supported range.</td>
    </tr>
      <tr>
      <td>quantModeOptional is not within the specified value range.</td>
    </tr>
  </tbody></table>

## aclnnSwiGluQuant

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device side. Obtained by the first-phase API aclnnSwiGluQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator calculation process.</td>
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

- deterministic computation:
  - aclnnSwiGluQuant defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swi_glu_quant.h"

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
  // Boilerplate for resource initialization
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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate strides of the contiguous tensor.
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
  // 1. (Boilerplate) Device/stream initialization. Refer to the acl API.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> xShape = {3, 32};
  std::vector<int64_t> smoothScalesShape = {2, 16};
  std::vector<int64_t> groupIndexShape = {2};
  std::vector<int64_t> outShape = {3, 16};
  std::vector<int64_t> scaleShape = {3};
  void* xDeviceAddr = nullptr;
  void* smoothScalesDeviceAddr = nullptr;
  void* groupIndexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* smoothScales = nullptr;
  aclTensor* groupIndex = nullptr;
  aclTensor* out = nullptr;
  aclTensor* scale = nullptr;
  std::vector<float> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                  43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                  63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
                                  83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
  std::vector<float> smoothScalesHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int32_t> groupIndexHostData = {1, 3};
  std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> scaleHostData = {0, 0, 0};

  // Create the x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the smoothScales aclTensor.
  ret = CreateAclTensor(smoothScalesHostData, smoothScalesShape, &smoothScalesDeviceAddr, aclDataType::ACL_FLOAT, &smoothScales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the groupIndex aclTensor.
  ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT32, &groupIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the scale aclTensor.
  ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnSwiGluQuant first-phase API.
  ret = aclnnSwiGluQuantGetWorkspaceSize(x, smoothScales, nullptr, groupIndex, false, "dynamic", out, scale, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwiGluQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnSwiGluQuant second-phase API.
  ret = aclnnSwiGluQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwiGluQuant failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Change it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
  // 6. Release aclTensor and aclScalar. Change it based on the specific API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(smoothScales);
  aclDestroyTensor(groupIndex);
  aclDestroyTensor(out);
  aclDestroyTensor(scale);
  // 7. Release device resources. Change it based on the specific API definition.
  aclrtFree(xDeviceAddr);
  aclrtFree(smoothScalesDeviceAddr);
  aclrtFree(groupIndexDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(scaleDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
