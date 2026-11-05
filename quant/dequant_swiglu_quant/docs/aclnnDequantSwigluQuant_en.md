# aclnnDequantSwigluQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:11:11.730Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
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

- Description: Adds dequant and quant operations before and after the Swish Gated Linear Unit activation function to implement DequantSwigluQuant computation on x.
- Formula:

  $$
  dequantOut = Dequant(x, weightScaleOptional, activationScaleOptional, biasOptional)
  $$

  $$
  swigluOut = Swiglu(dequantOut)=Swish(A)*B
  $$

  $$
  out = Quant(swigluOut, quantScaleOptional, quantOffsetOptional)
  $$

  where A represents the first half of dequantOut, and B represents the second half of dequantOut.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnDequantSwigluQuantGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnDequantSwigluQuant" API to execute the computation.

```Cpp
aclnnStatus aclnnDequantSwigluQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weightScaleOptional,
    const aclTensor *activationScaleOptional,
    const aclTensor *biasOptional,
    const aclTensor *quantScaleOptional,
    const aclTensor *quantOffsetOptional,
    const aclTensor *groupIndexOptional,
    bool             activateLeft,
    char            *quantModeOptional,
    const aclTensor *yOut,
    const aclTensor *scaleOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDequantSwigluQuant(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnDequantSwigluQuantGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 251px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 420px">
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
      <td>The shape is (N...,H). The last dimension must be a multiple of 2, and the number of dimensions of x must be greater than 1.</td>
      <td>FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>x</td>
    </tr>
     <tr>
      <td>weightScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Dequantization scale of weight, weightScaleOptional in the formula.</td>
      <td><ul><li>Optional parameter. Supports passing a null pointer.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>x</td>
     </tr>
    <tr>
      <td>activationScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Dequant scale of the activation function, activationScaleOptional in the formula.</td>
      <td><ul><li>Optional parameter. Supports passing a null pointer.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
      <tr>
      <td>biasOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Matmul bias, which is biasOptional in the formula.</td>
      <td><ul><li>The shape supports 1 or 2 dimensions, represented as [H] or [groupNum, H], and the value of H must keep consistent with the last dimension of x. When groupIndexOptional is a null pointer, the shape is [H]; when groupIndexOptional is not a null pointer, the shape is [groupNum, H].</li><li>Optional parameter. Supports passing a null pointer.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
       <tr>
      <td>quantScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Scale for quantization, quantScaleOptional in the formula.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>x</td>
    </tr>
       <tr>
      <td>quantOffsetOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Quantization offset, quantOffsetOptional in the formula.</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>x</td>
    </tr>
      <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input</td>
      <td>group_index required for MoE grouping.</td>
      <td><ul><li>Optional parameter, supports passing null.</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
      <tr>
      <td>activateLeft(bool)</td>
      <td>Input</td>
      <td>Indicates whether to apply swiglu activation to the left half of the input.</td>
      <td>When the value is false, activation is applied to the right half of the input.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantModeOptional (char*)</td>
      <td>Input</td>
      <td>Specifies whether to use dynamic quantization or static quantization.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>-</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>x</td>
    </tr>
    <tr>
      <td>scaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>-</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>x</td>
    </tr>
       <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the required Device side-applied workspace size.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - In the parameter weightScaleOptional, shape is represented as [H], and the value of H keeps consistent with the last dimension of x.
    - In the parameter activationScaleOptional, shape is [N..., 1], the last dimension is 1, and other dimensions keep consistent with x.
    - In the quantScaleOptional parameter, the shape validation rules are determined by whether groupIndexOptional is a null pointer and by the value of quantModeOptional (H is the last dimension of x, and H must be an even number):
      - When groupIndexOptional is a null pointer, the rules are the same for static and dynamic: a null pointer is supported; if not null, the number of elements is validated and must be H/2 (per channel) or 1 (per tensor), with no limit on the number of dimensions.
      - When groupIndexOptional is not a null pointer (groupNum is the first dimension of groupIndexOptional):
        - static: mandatory. The shape is [groupNum] (one quantization parameter per group) or [groupNum, H/2] (per channel per group). quantOffsetOptional must follow the same shape rules.
        - dynamic: optional. If not null, the shape is [H/2] or [1, H/2] (when groupNum is 1), or [groupNum, H/2] (when groupNum is greater than 1, the first dimension must be groupNum).
    - In the quantOffsetOptional parameter, its constraints are also determined by whether groupIndexOptional is a null pointer and by the value of quantModeOptional (H is the last dimension of x):
      - When groupIndexOptional is a null pointer:
        - static: if quantScaleOptional is a null pointer, this parameter does not take effect, and passing a null pointer is recommended; if quantScaleOptional is not a null pointer, this parameter is mandatory and its data type must be FLOAT. The number of elements is validated and must be H/2 (per channel) or 1 (per tensor), with no limit on the number of dimensions.
        - dynamic: this parameter does not take effect, and passing a null pointer is recommended.
      - When groupIndexOptional is not a null pointer (groupNum is the first dimension of groupIndexOptional):
        - static: mandatory, and the data type and shape must be the same as those of quantScaleOptional.
        - dynamic: must be a null pointer.
    - In the parameter quantModeOptional, "dynamic" and "static" are supported.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:
    - In the weightScaleOptional parameter, the shape is represented as [H] or [groupNum, H], and the value of H must keep consistent with the last dimension of x. When groupIndexOptional is a null pointer, the shape is [H] or [1, H]; when groupIndexOptional is not a null pointer, the shape is [groupNum, H].
    - In the activationScaleOptional parameter, the shape is [N,...]. The number of dimensions is no more than 7 and no less than 1, one less than that of x, and the shape is the same as that of x in corresponding dimensions.
    - In the quantScaleOptional parameter, only FLOAT is supported, and it is only supported in the use case where quantModeOptional is dynamic. When quantModeOptional is dynamic, the shape is represented as [H/2] or [groupNum, H/2]. When groupIndexOptional is a null pointer, the shape is [H/2]; when groupIndexOptional is not a null pointer, the shape is [groupNum, H/2].
    - In the quantOffsetOptional parameter, this parameter is temporarily not supported.
    - In the groupIndexOptional parameter, only INT64 is supported. The shape is [groupNum], and groupNum must be greater than or equal to 1. Optional parameter, supports passing a null pointer.
    - In the quantModeOptional parameter, only "dynamic" is supported. Passing a null pointer is supported, in which case "static" is used by default.
    - In the yOut parameter, the shape is [N...,H/2], and the last dimension of yOut must not exceed 5120.
    - In the scaleOut parameter, the shape is [N,...]. The number of dimensions is no more than 7 and no less than 1, and the shape is the same as that of yOut excluding the last dimension.
  <!-- end id8 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 316px">
  <col style="width: 111px">
  <col style="width: 723px">
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
    <td>x, yOut, or scaleOut is a null pointer.</td>
  </tr>
  <tr>
    <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="4">161002</td>
    <td>The data type of the input or output is not in the supported range.</td>
  </tr>
  <tr>
    <td>The number of dimensions of the input or output is not in the supported range.</td>
  </tr>
  <tr>
    <td>The shape of the input or output does not meet the constraint requirements.</td>
  </tr>
  <tr>
    <td>The input values do not meet the requirements.</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_INNER_TILING_ERROR</td>
    <td>561002</td>
    <td>The tensor memory size of the input exceeds the upper limit.</td>
  </tr>
  </tbody>
  </table>

## aclnnDequantSwigluQuant

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 167px">
  <col style="width: 123px">
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
      <td>Workspace memory address side-applied on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace side-applied on the Device, obtained by the first-phase API aclnnDequantSwigluQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator calculation process.</td>
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

- deterministic computation:
  - aclnnDequantSwigluQuant defaults to a deterministic implementation.

<!-- npu="A3,910b" id9 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - The last dimension of x must be a multiple of 2, and the dimension of x must be greater than 1.
  - The operator imposes an upper limit on the memory size of input tensors. Verification formula: weightScaleOptional tensor memory size + biasOptional tensor memory size + quantScaleOptional tensor memory size + quantOffsetOptional tensor memory size + (activationScaleOptional tensor memory size + scaleOut tensor memory size) / 40 + x tensor last dimension H memory size * 10 < 192 KB.
<!-- end id9 -->
<!-- npu="950" id10 -->
- <term>Ascend 950PR&950DT products</term>:
  - The last dimension of input x must be a multiple of 2, and the number of dimensions of x must be greater than 1.
  - When the data type of input x is INT32, weightScaleOptional cannot be null. When the data type of input x is not INT32, weightScaleOptional is not allowed and a null pointer must be passed.
  - When the data type of input x is not INT32, activationScaleOptional is not allowed and a null pointer must be passed.
  - When the data type of input x is not INT32, biasOptional is not allowed and a null pointer must be passed.
  - The trailing axis of output yOut must not exceed 5120.
<!-- end id10 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```C++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dequant_swiglu_quant.h"

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
  // (Fixed writing) Initialize AscendCL.
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> xShape = {2, 32};
  std::vector<int64_t> scaleShape = {1};
  std::vector<int64_t> offsetShape = {1};
  std::vector<int64_t> outShape = {2, 16};
  std::vector<int64_t> scaleOutShape = {2};
  void* xDeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  void* offsetDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* scaleOutDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* scale = nullptr;
  aclTensor* offset = nullptr;
  aclTensor* out = nullptr;
  aclTensor* scaleOut = nullptr;
  std::vector<aclFloat16> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                       23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                       43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  std::vector<float> scaleHostData = {1};
  std::vector<float> offsetHostData = {1};
  std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> scaleOutHostData = {0, 0};

  // Create an x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a scale aclTensor.
  ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an offset aclTensor.
  ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create scaleOut aclTensor.
  ret = CreateAclTensor(scaleOutHostData, scaleOutShape, &scaleOutDeviceAddr, aclDataType::ACL_FLOAT, &scaleOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnDequantSwigluQuant.
  ret = aclnnDequantSwigluQuantGetWorkspaceSize(x, nullptr, nullptr, nullptr, scale, offset, nullptr, false, "static", out, scaleOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSwigluQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnDequantSwigluQuant.
  ret = aclnnDequantSwigluQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSwigluQuant failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(scale);
  aclDestroyTensor(offset);
  aclDestroyTensor(out);
  aclDestroyTensor(scaleOut);
  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(xDeviceAddr);
  aclrtFree(scaleDeviceAddr);
  aclrtFree(offsetDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(scaleOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
