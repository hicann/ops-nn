# aclnnFusedMatmul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:05:11.623Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supports the ND scenario where fusedOpType is relu (FLOAT16, BFLOAT16, and FLOAT32; FLOAT32 supports KEEP_DTYPE and HF32)
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supports the ND scenario where fusedOpType is relu (FLOAT16, BFLOAT16, and FLOAT32; FLOAT32 supports KEEP_DTYPE and HF32)
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

- Description: Fuses matrix multiplication with general vector computation.
- Formula:

  $$
  y = OP((x1 @ x2 + bias), x3)
  $$

  The OP type is defined by the fusedOpType input and supports the following:

  add operation:

  $$
  y=(x1 @ x2 + bias) + x3
  $$

  mul operation:

  $$
  y=(x1 @ x2 + bias) ∗ x3
  $$

  gelu_tanh operation:

  $$
  y = gelu\_tanh(x1 @ x2 + bias)
  $$

  gelu_erf operation:

  $$
  y = gelu\_erf(x1 @ x2 + bias)
  $$

  relu operation:

  $$
  y = relu(x1 @ x2 + bias)
  $$

  16cast32 operation:

  $$
  y = cast\_float32(x1 @ x2 + bias)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call "aclnnFusedMatmulGetWorkspaceSize" to obtain the workspace size required for the computation and the executor containing the operator computation flow, and then call "aclnnFusedMatmul" to perform the computation.

```cpp
aclnnStatus aclnnFusedMatmulGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *bias,
  const aclTensor *x3,
  const char      *fusedOpType,
  int8_t           cubeMathType,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnFusedMatmul(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnFusedMatmulGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
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
        <th>Dimension</th>
        <th>Non-contiguous</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x1</td>
        <td>Input</td>
        <td>Represents the first matrix of the matrix multiplication, corresponding to x1 in the formula.</td>
        <td><li>The data type must satisfy the data type deduction rules with x2 (see also <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2-6 (fusedOpType is "" or "relu"); 2-3 (fusedOpType is "add" or "mul"); 2 (other values)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>Input</td>
        <td>Represents the second matrix of the matrix multiplication, corresponding to x2 in the formula.</td>
        <td><li>The data type must comply with the data type derivation rule with x1 (see also <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li></td>
        <td>The data type remains consistent with x1.</td>
        <td>ND</td>
        <td>2-6 (when fusedOpType is "" or "relu"); 2-3 (when fusedOpType is "add" or "mul"); 2 (other values)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>Input</td>
        <td>Bias term, corresponding to bias in the corresponding formula.</td>
        <td><li>This parameter takes effect only when fusedOpType is "", "16cast32", "relu", "add", "mul", "gelu_erf", or "gelu_tanh". Pass a null pointer in other cases.</li></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x3</td>
        <td>Input</td>
        <td>The second matrix of the fused operation, corresponding to x3 in the formula.</td>
        <td>Takes effect only when fusedOpType is "add" or "mul". Otherwise, pass in a null pointer.</td>
        <td>The data type must remain consistent with x1.</td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td>Represents the output matrix of the computation, that is, y in the corresponding formula.</td>
        <td><li>The data type must remain consistent with the deduced data types of x1 and x2 (see also <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>); when fusedOpType is "16cast32", the data type of y is fixed to FLOAT32.</li></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2-6 (fusedOpType is "" or "relu"); 2-3 (fusedOpType is "add" or "mul"); 2 (other values)</td>
        <td>√</td>
      </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Specifies the computation logic of the Cube unit.</td>
      <td>If the data types of the input have a deduction relationship, this parameter processes the mutually derived data type by default. The supported enumeration values are as follows:<ul>
        <li>0: KEEP_DTYPE, keeps the input data type for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION, allows the input data to be computed at reduced precision.</li>
        <li>2: USE_FP16, allows the input to be reduced to FLOAT16 for computation.</li>
        <li>3: USE_HF32, allows the input to be reduced to data type HFLOAT32 for computation.</li>
        <li>4: USE_FP32_ADD, supports computation with the high-precision method.</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
        <td>fusedOpType</td>
        <td>Input</td>
        <td>Specifies the fusion mode supported by the Matmul operator, corresponding to OP in the formula.</td>
        <td><li>The fused mode value must be one of "" (indicating no fusion), "16cast32", "add", "mul", "gelu_erf", "gelu_tanh", and "relu".</li></td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the workspace that the user needs to allocate on the Device side.</td>
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

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:
    - cubeMathType=1: this option is not supported when the input data type is FLOAT32; no processing is performed for other input data types;
    - cubeMathType=2: This option is not supported when the input data type is BFLOAT16.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation. This option is not supported when the input is of other data types.
    - cubeMathType=4: the high-precision computation is used only when all of the following conditions are met:
      - fusedOpType is set to "add" or "mul";
      - x1 and x2 must have the same number of dimensions, and both must be 2D or both must be 3D;
      - x1, x2, and x3 must have exactly the same data type, and all three must be BFLOAT16 or all three must be FLOAT16.

  <!-- end id7 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
    </colgroup>
    <thead>
      <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
        <td rowspan="2">161001</td>
        <td>x1, x2, or y is a null pointer.</td>
      </tr>
      <tr>
        <td>When fusedOpType is add or mul, x3 is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>The data type of x1 or x2 is not within the supported range.</td>
      </tr>
      <tr>
        <td>The data format of x1, x2, or y is not within the supported range.</td>
      </tr>
      <tr>
        <td>The dimensions of x1, x2, or y do not meet the requirements: when fusedOpType is "" or "relu", 2-6 dimensions are supported; when fusedOpType is "add" or "mul", 2-3 dimensions are supported; when fusedOpType is "16cast32", two dimensions or three dimensions with a batch axis of 1 are supported; for other values, two dimensions are supported.</td>
      </tr>
      <tr>
        <td>When fusedOpType is "add" or "mul" and the shape of x3 does not meet requirements: if x3 is 2D, Shape[-2] and Shape[-1] must remain consistent with Shape[-2] and Shape[-1] of y; if x3 is 3D, Shape[-2] and Shape[-1] must remain consistent with Shape[-2] and Shape[-1] of y, and the batch axis must be consistent with that of y or be 1.</td>
      </tr>
      <tr>
        <td>The passed-in fusedOpType is not one of "", "16cast32", "add", "mul", "gelu_tanh", "gelu_erf", or "relu".</td>
      </tr>
      <tr>
        <td>The data types of x1 and x2 cannot be mutually derived.</td>
      </tr>
      <tr>
        <td>When the passed-in fusedOpType is one of "", "add", "mul", or "relu" and the input data type is float32, cubeMathType supports only 3.</td>
      </tr>
  </tbody></table>

## aclnnFusedMatmul

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
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
        <td>Size of the workspace allocated on the Device, obtained by calling the first-phase API aclnnFusedMatmulGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream on which the task is executed.</td>
      </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- GELU (gelu_erf, gelu_tanh) supports an optional bias. The computation order is to first compute matmul and add bias, then perform GELU. The data type of bias must be the same as that of the matrix input or be FLOAT32, and its shape must be (N,) or (1, N). When bias is passed in, the K axis of matmul must be greater than 0.

- On <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, only fusedOpType "relu" is supported. Inputs and outputs are ND, and the data type of x1, x2, and y is FLOAT16, BFLOAT16, or FLOAT32. FLOAT32 supports cubeMathType=KEEP_DTYPE (true FP32) and cubeMathType=3 (HF32). The formula is y = relu(x1 @ x2 + bias). Rank 2 uses MatMulV3. Ranks 3 to 6 with matching batch dimensions use BatchMatMulV3. The relu case that shares x2 and merges the batch axis on <term>Ascend 950PR&950DT products</term> is not supported.

- Deterministic computation:

  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnFusedMatmul defaults to a deterministic implementation.
  <!-- end id8 -->
  <!-- npu="A3" id9 -->
  - <term>Atlas A3 products</term>: When fusedOpType is relu, the implementation is deterministic by default.
  <!-- end id9 -->
  <!-- npu="910b" id10 -->
  - <term>Atlas A2 products</term>: When fusedOpType is relu, the implementation is deterministic by default.
  <!-- end id10 -->

- When fusedOpType is "gelu_erf" or "gelu_tanh", the data types of x1 and x2 must be BFLOAT16 or FLOAT16, and only 2D x1, x2, and y are supported. When fusedOpType is "" or "relu", the data types of x1 and x2 must be FLOAT32 (only cubeMathType 3 is supported), BFLOAT16, or FLOAT16. When fusedOpType is "16cast32", the data types of x1 and x2 must be BFLOAT16 or FLOAT16, and 2D x1, x2, and y, or 3D tensors with a batch axis of 1, are supported. When fusedOpType is "add" or "mul", the data types of x1, x2, and x3 must be FLOAT32 (only cubeMathType 3 is supported), BFLOAT16, or FLOAT16.
- When fusedOpType is "", "relu", "add", or "mul", in multi-dimensional use cases where broadcast is not applicable, the batch dimensions must remain consistent.
- When fusedOpType is "add" or "mul", in the BMM (3D) use case, x1, x2, and y support 3D tensors; x3 supports 2D or 3D tensors. A 2D x3 can be broadcast as a matrix for 3D output, and the batch axis of a 3D x3 must remain consistent with that of y or be 1.
- When fusedOpType is "16cast32", the data type of output y must be FLOAT32.
- When x1 or x2 is an empty tensor, fusedOpType values "", "relu", "gelu_erf", "gelu_tanh", "add", and "mul" are supported, while "16cast32" is not supported.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_matmul.h"

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
  // Boilerplate: resource initialization
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
  // Call aclrtMalloc to request device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of contiguous tensors.
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
  // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Custom construction is required according to the API.
  std::vector<int64_t> xShape = {16, 32};
  std::vector<int64_t> x2Shape = {32, 16};
  std::vector<int64_t> x3Shape = {16, 16};
  std::vector<int64_t> yShape = {16, 16};
  void* xDeviceAddr = nullptr;
  void* x2DeviceAddr = nullptr;
  void* x3DeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* x2 = nullptr;
  aclTensor* x3 = nullptr;
  aclTensor* y = nullptr;
  std::vector<float> xHostData(512, 1);
  std::vector<float> x2HostData(512, 1);
  std::vector<float> x3HostData(256, 1);
  std::vector<float> yHostData(256, 0);
  // Create the x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the x2 aclTensor.
  ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the x3 aclTensor.
  ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the y aclTensor.
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  int8_t cubeMathType = 0;
  const char* fusedOpType = "add";
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  // Call the first-phase API of aclnnFusedMatmul.
  ret = aclnnFusedMatmulGetWorkspaceSize(x, x2, nullptr, x3, fusedOpType, cubeMathType, y, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnFusedMatmul.
  ret = aclnnFusedMatmul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedMatmul failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. Modify it according to the specific API definition.
  auto size = GetShapeSize(yShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it according to the specific API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(x2);
  aclDestroyTensor(x3);
  aclDestroyTensor(y);

  // 7. Release device resources. Modify it according to the specific API definition.
  aclrtFree(xDeviceAddr);
  aclrtFree(x2DeviceAddr);
  aclrtFree(x3DeviceAddr);
  aclrtFree(yDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
