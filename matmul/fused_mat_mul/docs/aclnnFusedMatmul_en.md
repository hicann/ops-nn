# aclnnFusedMatmul

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Function: fuses matrix multiplication and general vector computation.
- Formula:

  $$
  y = OP((x1 @ x2 + bias), x3)
  $$

  The OP type is defined by the input fusedOpType. The following types are supported:

  Add operation:

  $$
  y=(x1 @ x2 + bias) + x3
  $$

  Mul operation:

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

  Relu operation:

  $$
  y = relu(x1 @ x2 + bias)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. You must call aclnnFusedMatmulGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call aclnnFusedMatmul to perform the computation.

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

- **Parameters**
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
    <thread>
      <tr>
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension</th>
        <th>Discontinuous</th>
      </tr></thread>
    <tbody>
      <tr>
        <td>x1</td>
        <td>Input</td>
        <td>First matrix for matrix multiplication, corresponding to x1 in the formula.</td>
        <td>Its data type and the data type of x2 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>Input</td>
        <td>Second matrix of matrix multiplication, corresponding to x2 in the formula.</td>
        <td>Its data type and the data type of x1 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</td>
        <td>The data type is the same as that of x1.</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>Input</td>
        <td>Bias, corresponding to bias in the formula.</td>
        <td>This parameter is valid only when fusedOpType is set to "", "relu", "add", or "mul". In other cases, pass a null pointer.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x3</td>
        <td>Input</td>
        <td>Second matrix of the fusion operation, corresponding to x3 in the formula.</td>
        <td>-</td>
        <td>The data type is the same as that of x1.</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td>Indicates the output matrix of the computation, corresponding to y in the formula.</td>
        <td>Its data type and the data types of x1 and x2 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Computation logic of the Cube unit.</td>
      <td>If the input data types can be deduced from each other, this parameter processes the deduced data type by default. The supported enumerated values are as follows:<ul>
        <li>0: KEEP_DTYPE. The input data type is retained for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION. The input data can be computed with reduced precision.</li>
        <li>2: USE_FP16. The input data can be downgraded to FLOAT16 for computation.</li>
        <li>3: USE_HF32. The input data can be downgraded to HFLOAT32 for computation.</li>
        <li>4: USE_FP32_ADD, indicating that the computation can be performed in high-precision mode.</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
        <td>fusedOpType</td>
        <td>Input</td>
        <td>Indicates the fusion mode supported by the Matmul operator, corresponding to OP in the formula.</td>
        <td>The value of the fusion mode must be one of the following: "" (indicating that fusion is not performed), "add", "mul", "gelu_erf", "gelu_tanh", or "relu".</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
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

  - Ascend 950PR/Ascend 950DT:
    - When **cubeMathType** is set to **1**, if the input data type is FLOAT32, it is converted to HFLOAT32 for computation. If the input data type is not FLOAT32, no processing is performed.
    - When **cubeMathType** is set to **2**, this option is not supported if the input data type is BFLOAT16.
    - When **cubeMathType** is set to **3**, if the input data type is FLOAT32, it is converted to HFLOAT32 for computation. If the input data type is not FLOAT32, this option is not supported.
    - When **cubeMathType** is set to **4**, no processing is performed.

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
    </colgroup>
    <thread>
      <tr>
        <th>Return</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thread>
    <tbody>
      <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_NULLPTR</td>
        <td rowspan="3">161001</td>
        <td>The input x1, x2, and y are null pointers.</td>
      </tr>
      <tr>
        <td>When fusedOpType is set to add or mul, the input x3 is a null pointer.</td>
      </tr>
      <tr>
        <td>When fusedOpType is set to gelu_tanh or gelu_erf, the input bias is not a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>The data types of x1 and x2 are not supported.</td>
      </tr>
      <tr>
        <td>The data format of x1, x2, or y is not supported.</td>
      </tr>
      <tr>
        <td>The dimensions of x1 and x2 are not two-dimensional.</td>
      </tr>
      <tr>
        <td>When fusedOpType is set to add or mul, the shape of x3 is inconsistent with the output shape.</td>
      </tr>
      <tr>
        <td>The input fusedOpType is not one of "", "add", "mul", "gelu_tanh", "gelu_erf", and "relu".</td>
      </tr>
      <tr>
        <td>Data type deduction cannot be performed for x1 and x2.</td>
      </tr>
      <tr>
        <td>When the input fusedOpType is one of "", "add", "mul", and "relu" and the input data type is float32, cubeMathType supports only 3.</td>
      </tr>
  </tbody></table>

## aclnnFusedMatmul

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
    </colgroup>
    <thread>
      <tr>
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr></thread>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Memory address of the workspace to be allocated on the device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnFusedMatmulGetWorkspaceSize.</td>
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
  </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic description:
  - For <term>Atlas training products</term> and <term>Atlas inference products</term>, aclnnFusedMatmul is implemented in a deterministic manner by default.

- When fusedOpType is set to "gelu_erf" or "gelu_tanh", the data types of x1, x2, and x3 must be BFLOAT16 and FLOAT16. When fusedOpType is set to "", "relu", "add", or "mul", the data types of x1, x2, and x3 must be FLOAT32 (cubeMathType supports only 3), BFLOAT16, or FLOAT16.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
  // Create an x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an x2 aclTensor.
  ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an x3 aclTensor.
  ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a y aclTensor.
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  int8_t cubeMathType = 0;
  const char* fusedOpType = "add";
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  // Call the first part of the aclnnFusedMatmul API.
  ret = aclnnFusedMatmulGetWorkspaceSize(x, x2, nullptr, x3, fusedOpType, cubeMathType, y, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second part of the aclnnFusedMatmul API.
  ret = aclnnFusedMatmul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedMatmul failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(yShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
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
