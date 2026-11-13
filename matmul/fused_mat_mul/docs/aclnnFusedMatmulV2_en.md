# aclnnFusedMatmulV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:06:51.857Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: Fuses matrix multiplication and general vector computation. Compared with [aclnnFusedMatmul](aclnnFusedMatmul.md), this API adds the optional inputs alphaOptional and betaOptional, which scale the matrix multiplication result and x3, respectively. Both parameters default to 1.0 when a null pointer is passed in. Choose an appropriate API based on the actual use case.
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

  When aclnnFusedMatmulV2 is called with a non-default alphaOptional or betaOptional and fusedOpType is "add":

  $$
  y = alphaOptional * (x1 @ x2) + betaOptional * x3
  $$

  This use case does not support bias. The operator completes the computation using the "scale_add" fusion mode internally.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call aclnnFusedMatmulV2GetWorkspaceSize to obtain the workspace size required for the computation and the executor that contains the operator computation process. Then call aclnnFusedMatmulV2 to execute the computation.

```cpp
aclnnStatus aclnnFusedMatmulV2GetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *bias,
  const aclTensor *x3,
  const aclScalar *alphaOptional,
  const aclScalar *betaOptional,
  const char      *fusedOpType,
  int8_t           cubeMathType,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnFusedMatmulV2(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnFusedMatmulV2GetWorkspaceSize

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
        <th>Dimensions</th>
        <th>Non-contiguous</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x1</td>
        <td>Input</td>
        <td>The first matrix for matrix multiplication, which is x1 in the corresponding formula.</td>
        <td><li>The data type must satisfy the data type deduction rule with x2 (see also <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2-6 (when fusedOpType is "" or "relu"); 2-3 (when fusedOpType is "add" or "mul"); 2 (for other values)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>Input</td>
        <td>The second matrix for matrix multiplication, which is x2 in the corresponding formula.</td>
        <td><li>The data type must satisfy the data type deduction rules with x1 (see also <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>).</li></td>
        <td>The data type is the same as that of x1</td>
        <td>ND</td>
        <td>2-6 (when fusedOpType is "" or "relu"); 2-3 (when fusedOpType is "add" or "mul"); 2 (other values)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>Input</td>
        <td>Represents the bias term, corresponding to bias in the formula.</td>
        <td><li>Takes effect only when fusedOpType is "", "16cast32", "relu", "add", "mul", "gelu_erf", or "gelu_tanh". For other cases, pass in a null pointer.</li></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x3</td>
        <td>Input</td>
        <td>Indicates the second matrix of the fusion operation, corresponding to x3 in the formula.</td>
        <td>Takes effect only when fusedOpType is "add" or "mul". In other cases, pass a null pointer.</td>
        <td>Same as x1</td>
        <td>ND</td>
        <td>2-3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>alphaOptional</td>
        <td>Input</td>
        <td>Optional scaling coefficient used to scale the matmul result.</td>
        <td><li>If a null pointer is passed in, the value is 1.0.</li></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>betaOptional</td>
        <td>Input</td>
        <td>Optional scaling coefficient, used to scale x3.</td>
        <td><li>Defaults to 1.0 when a null pointer is passed.</li></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>fusedOpType</td>
        <td>Input</td>
        <td>Represents the fusion mode supported by the Matmul operator, corresponding to OP in the formula.</td>
        <td><li>The fusion mode must be one of "" (no fusion), "16cast32", "add", "mul", "gelu_erf", "gelu_tanh", and "relu".</li>
        <li>"scale_add" is a fusion mode used internally by the operator and cannot be passed in directly by users. When fusedOpType is "add" and alphaOptional or betaOptional is not 1, the operator internally uses "scale_add". Therefore, this value may appear in runtime logs.</li></td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>cubeMathType</td>
        <td>Input</td>
        <td>Specifies the computation logic of the Cube unit.</td>
        <td>If the input data types have a deduction relationship, this parameter processes the data type after deduction by default. Supported enumeration values are as follows: <ul>
          <li>0: KEEP_DTYPE, keeps the input data type for computation.</li>
          <li>1: ALLOW_FP32_DOWN_PRECISION, supports computing the input data at reduced precision.</li>
          <li>2: USE_FP16, supports computing the input at FLOAT16 precision.</li>
          <li>3: USE_HF32, supports computing the input at the HFLOAT32 data type.</li>
          <li>4: USE_FP32_ADD, supports computing in high-precision mode.</li></ul>
        </td>
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td>Represents the output matrix of the computation, corresponding to y in the formula.</td>
        <td><li>The data type must be the same as the deduced data type of x1 and x2 (see also <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>); when fusedOpType is "16cast32", the data type of y is fixed to FLOAT32.</li></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2-6 (when fusedOpType is "" or "relu"); 2-3 (when fusedOpType is "add" or "mul"); 2 (other values)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that users need to allocate on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the computation flow of the operator.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody></table>

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:
    - cubeMathType=1: This option is not supported when the input data type is FLOAT32; no processing is performed when the input is of other data types;
    - cubeMathType=2: This option is not supported when the input data type is BFLOAT16.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation. This option is not supported when the input is of other data types.
    - cubeMathType=4: The high-precision method is used only when all of the following conditions are met:
      - fusedOpType is "add" or "mul";
      - x1 and x2 must have the same number of dimensions, both 2D or both 3D;
      - x1, x2, and x3 must have the same data type, all BFLOAT16 or all FLOAT16.

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
        <td>When fusedOpType is add or mul, the input x3 is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="9">161002</td>
        <td>The data types of x1 and x2 are not within the supported range.</td>
      </tr>
      <tr>
        <td>The data formats of x1, x2, or y are not within the supported range.</td>
      </tr>
      <tr>
        <td>The dimensions of x1, x2, or y do not meet requirements: 2-6 dimensions are supported when fusedOpType is "" or "relu"; 2-3 dimensions are supported when fusedOpType is "add" or "mul"; 2 dimensions, or 3 dimensions with a batch axis of 1, are supported when fusedOpType is "16cast32"; 2 dimensions are supported for other values.</td>
      </tr>
      <tr>
        <td>The K-axis lengths of x1 and x2 used for matrix multiplication are inconsistent.</td>
      </tr>
      <tr>
        <td>When fusedOpType is "add" or "mul", the shape of x3 does not meet requirements: if x3 is 2-dimensional, Shape[-2] and Shape[-1] must be the same as Shape[-2] and Shape[-1] of y; if x3 is 3-dimensional, Shape[-2] and Shape[-1] must be the same as Shape[-2] and Shape[-1] of y, and the batch axis must be the same as that of y or be 1.</td>
      </tr>
      <tr>
        <td>The input fusedOpType is not one of "", "16cast32", "add", "mul", "gelu_tanh", "gelu_erf", or "relu".</td>
      </tr>
      <tr>
        <td>Data type deduction cannot be performed for x1 and x2.</td>
      </tr>
      <tr>
        <td>When fusedOpType is one of "", "add", "mul", or "relu" and the input data type is FLOAT32, cubeMathType supports only 3.</td>
      </tr>
      <tr>
        <td>The data type of alphaOptional or betaOptional is out of the supported range, or the non-default scaling coefficient use case does not satisfy the constraints.</td>
      </tr>
  </tbody></table>

## aclnnFusedMatmulV2

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
        <td>The workspace memory address allocated on the Device side.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>The workspace size allocated on the Device side, obtained by the first-phase API aclnnFusedMatmulV2GetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the Stream for task execution.</td>
      </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- GELU (gelu_erf, gelu_tanh) supports an optional bias. The computation order is to compute the matrix multiplication and add bias first, and then perform GELU. The data type of bias must be the same as that of the matrix input or be FLOAT32, and the shape must be (N,) or (1, N); when bias is passed in, the K axis of the matrix multiplication must be greater than 0.

- Deterministic computation description:

  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnFusedMatmulV2 defaults to a deterministic implementation.

  <!-- end id8 -->

- When fusedOpType is "gelu_erf" or "gelu_tanh", the data types of x1 and x2 must be BFLOAT16 or FLOAT16, and only 2-dimensional x1, x2, and y are supported. When fusedOpType is "" or "relu", the data types of x1 and x2 must be FLOAT32 (cubeMathType supports only 3), BFLOAT16, or FLOAT16. When fusedOpType is "16cast32", the data types of x1 and x2 must be BFLOAT16 or FLOAT16, and x1, x2, and y can be 2-dimensional or 3-dimensional with the batch axis being 1. When fusedOpType is "add" or "mul", the data types of x1, x2, and x3 must be FLOAT32 (cubeMathType supports only 3), BFLOAT16, or FLOAT16.
- When fusedOpType is "", "relu", "add", or "mul", in multi-dimensional use cases where broadcasting is not applicable, the batch dimensions must be the same.
- When fusedOpType is "add" or "mul", in the BMM (3-dimensional) use case, x1, x2, and y support 3 dimensions. x3 supports 2-3 dimensions. A 2-dimensional x3 can be broadcast as a matrix for a 3-dimensional output. For a 3-dimensional x3, its batch axis must be the same as that of y or be 1.
- When fusedOpType is "16cast32", the data type of the output y must be FLOAT32.
- If x1 or x2 is an empty tensor, the values "", "relu", "gelu_erf", "gelu_tanh", "add", and "mul" of fusedOpType are supported, but the value "16cast32" is not supported.
- In aclnnFusedMatmulV2, if alphaOptional or betaOptional is not 1, only the 3-dimensional "add" use case where B, M, N, and K are all greater than 0 is supported. This use case does not support bias, transposition, or batch axis broadcasting. The data types of x1, x2, x3, and y must all be FLOAT16, BFLOAT16 or FLOAT32 (FLOAT32 requires cubeMathType=3 and runs in HF32 mode).

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
  // Boilerplate code for resource initialization
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
  // Call aclrtMemcpy to copy host data to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute strides of a contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Customize the construction based on the API.
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
  // Create x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create x2 aclTensor.
  ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create x3 aclTensor.
  ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create y aclTensor.
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  int8_t cubeMathType = 0;
  const char* fusedOpType = "add";
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  // Call the first-phase API of aclnnFusedMatmulV2. The value is 1.0 if alphaOptional and betaOptional are null pointers.
  ret = aclnnFusedMatmulV2GetWorkspaceSize(x, x2, nullptr, x3, nullptr, nullptr, fusedOpType, cubeMathType, y,
                                            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnFusedMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnFusedMatmulV2.
  ret = aclnnFusedMatmulV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedMatmulV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code based on the API definition.
  auto size = GetShapeSize(yShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(x2);
  aclDestroyTensor(x3);
  aclDestroyTensor(y);

  // 7. Release device resources. Modify the code based on the API definition.
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
