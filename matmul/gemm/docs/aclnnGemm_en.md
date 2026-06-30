# aclnnGemm

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/gemm)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    √     |
| <term>Atlas training series products</term>                             |    √     |

## Function

- Description: Computes α multiplied by the product of A and B, and then adds the product of β and input C.
- Formula:
  - If transA is not zero, A is transposed before computation. Similarly, if transB is not zero, B is transposed before computation.

    $$
    out = α  (A @ B) + β  C
    $$

  - If both transA and transB are not zero, the formula is as follows:

    $$
    out = α  (A^T @ B^T) + βC
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnGemmGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnGemm** is called to perform computation.

```cpp
aclnnStatus aclnnGemmGetWorkspaceSize(
  const aclTensor *A, 
  const aclTensor *B, 
  const aclTensor *C, 
  float           alpha, 
  float           beta, 
  int64_t         transA, 
  int64_t         transB, 
  aclTensor       *out, 
  int8_t          cubeMathType, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```
```cpp
aclnnStatus aclnnGemm(
  void          *workspace, 
  uint64_t      workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream   stream)
```

## aclnnGemmGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1513px"><colgroup>
  <col style="width: 155px">
  <col style="width: 123px">
  <col style="width: 292px">
  <col style="width: 318px">
  <col style="width: 209px">
  <col style="width: 122px">
  <col style="width: 147px">
  <col style="width: 147px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Precaution</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>A</td>
      <td>Input</td>
      <td>Input A in the formula.</td>
      <td><ul><li>The data type must have a deduction relationship with those of C and B.</li>
      <li>The shape (or transposed shape) must meet the condition of multiplication with B.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>B</td>
      <td>Input</td>
      <td>Input B in the formula.</td>
      <td><ul><li>The data type must have a deduction relationship with those of C and A.</li>
      <li>The shape (or transposed shape) must meet the condition of multiplication with A.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>C</td>
      <td>Input</td>
      <td>Input C in the formula.</td>
      <td><ul><li>The data type must have a deduction relationship with the result of A multiplied by B.</li>
      <li>The shape must be the same as or meet the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with the result of A multiplied by B.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Input α in the formula, FLOAT type on the host, indicating the coefficient of the product of A and B.</td>
      <td>-</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Input β in the formula, FLOAT type on the host, indicating the coefficient of C.</td>
      <td>-</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transA</td>
      <td>Input</td>
      <td>Input transA in the formula, integer on the host, indicating whether matrix A needs to be transposed. If the value is not 0, matrix A, which is [K, M], needs to be transposed. If the value is 0, matrix A, which is [M, K], does not need to be transposed.</td>
      <td>-</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transB</td>
      <td>Input</td>
      <td>Input transB in the formula, integer on the host, indicating whether matrix B needs to be transposed. If the value is not 0, matrix B, which is [N, K], needs to be transposed. If the value is 0, matrix B, which is [K, N], does not need to be transposed.</td>
      <td>-</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>out in the formula. The data type must have a deduction relationship with that of C, and the shape must be the same as the result of A multiplied by B.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Determines the computation logic to be used by the Cube unit.</td>
      <td>If the input data types can be deduced from each other, this parameter processes the deduced data type by default. The supported enumerated values are as follows:<ul>
        <li>0: KEEP_DTYPE. The input data type is retained for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION. The input data type can be converted to reduce precision for computation.</li>
        <li>2: USE_FP16. The input data type can be converted to FLOAT16 for computation.</li>
        <li>3: USE_HF32. The input data can be downgraded to HFLOAT32 for computation.</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
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

  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>:
    - The data type of **A** can be FLOAT16 or FLOAT32.
    - The data type of **B** can be FLOAT16 or FLOAT32.
    - The data type of **C** can be FLOAT16 or FLOAT32.
    - The data type of **out** can be FLOAT16 or FLOAT32.
    - The data type BFLOAT16 is not supported.
    - When the input data type is FLOAT32, **cubeMathType** cannot be set to **0**.
    - When **cubeMathType** is set to **1**, if the input data type is FLOAT32, it is converted to FLOAT16 for computation. If the input data type is not FLOAT32, no processing is performed.
    - **cubeMathType** cannot be set to **3**.
  - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:
    - The data type of **A** can be BFLOAT16, FLOAT16, or FLOAT32.
    - The data type of **B** can be BFLOAT16, FLOAT16, or FLOAT32.
    - The data type of **C** can be BFLOAT16, FLOAT16, or FLOAT32.
    - The data type of **out** can be BFLOAT16, FLOAT16, or FLOAT32.
    - When **cubeMathType** is set to **1**, if the input data type is FLOAT32, it is converted to HFLOAT32 for computation. If the input data type is not FLOAT32, no processing is performed.
    - When **cubeMathType** is set to **2**, this option is not supported if the input data type is BFLOAT16.
    - When **cubeMathType** is set to **3**, if the input data type is FLOAT32, it is converted to HFLOAT32 for computation. If the input data type is not FLOAT32, this option is not supported.

- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).


  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed A, B, C, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type or format is not supported.</td>
    </tr>
    <tr>
      <td>A or B is not 2D, or k is not the same between [m, k] and [k, n] in the shape during computation.</td>
    </tr>
    <tr>
      <td>The broadcast operation cannot be performed between self and batch1@batch2.</td>
    </tr>
    <tr>
      <td>The computation result of C and AB does not meet the broadcast relationship.</td>
    </tr>
    <tr>
      <td>The shape of out is inconsistent with that after A is multiplied by B.</td>
    </tr>
    <tr>
      <td>The value of cubeMathType is invalid.</td>
    </tr>
  </tbody>
  </table>

## aclnnGemm

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnGemmGetWorkspaceSize.</td>
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

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic description:
  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: **aclnnGemm** defaults to a deterministic implementation. 
- <term>Atlas training series products</term> and <term>Atlas inference series products</term>: The Cube unit does not support FLOAT32 computation. The input data type FLOAT32 can be converted to FLOAT16 in the API for computation by setting **cubeMathType** to **1** (**ALLOW_FP32_DOWN_PRECISION**).

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/level2/aclnn_gemm.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // (Fixed writing) Initialize resources.
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external ACL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> AShape = {2, 2};
  std::vector<int64_t> BShape = {2, 2};
  std::vector<int64_t> CShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* ADeviceAddr = nullptr;
  void* BDeviceAddr = nullptr;
  void* CDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* A = nullptr;
  aclTensor* B = nullptr;
  aclTensor* C = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> AHostData = {1, 2, 1, 2};
  std::vector<float> BHostData = {1, 2, 1, 2};
  std::vector<float> CHostData = {1, 1, 1, 1};
  std::vector<float> outHostData = {0, 0, 0, 0};
  float alpha = 1.0f;
  float beta = 2.0f;
  int64_t transA = 0;
  int64_t transB = 0;
  int8_t cubeMathType = 1;

  // Create an A aclTensor.
  ret = CreateAclTensor(AHostData, AShape, &ADeviceAddr, aclDataType::ACL_FLOAT, &A);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a B aclTensor.
  ret = CreateAclTensor(BHostData, BShape, &BDeviceAddr, aclDataType::ACL_FLOAT, &B);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a C aclTensor.
  ret = CreateAclTensor(CHostData, CShape, &CDeviceAddr, aclDataType::ACL_FLOAT, &C);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGemm.
  ret = aclnnGemmGetWorkspaceSize(A, B, C, alpha, beta, transA, transB, out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnGemm.
  ret = aclnnGemm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemm failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(A);
  aclDestroyTensor(B);
  aclDestroyTensor(C);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(ADeviceAddr);
  aclrtFree(BDeviceAddr);
  aclrtFree(CDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
