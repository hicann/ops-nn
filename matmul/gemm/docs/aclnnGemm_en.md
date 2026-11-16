# aclnnGemm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:10:24.851Z -->

## Applicable Products

<!-- npu="950" id10 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id1 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->
<!-- npu="310b" id3 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Supported
<!-- end id5 -->

## Function

- Description: Computes α multiplied by the product of A and B, then summed with the product of β and input C.
- Formula:
  - If transA is non-zero, A is transposed before computation; similarly, if transB is non-zero, B is transposed.

    $$
    out = α  (A @ B) + β  C
    $$

  - If both transA and transB are non-zero, the formula is:

    $$
    out = α  (A^T @ B^T) + βC
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnGemmGetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size based on the calculation process, and then call the "aclnnGemm" API to execute the calculation.

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

- **Parameters**:
  <table style="undefined;table-layout: fixed; width: 1587px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 230px">
  <col style="width: 400px">
  <col style="width: 249px">
  <col style="width: 117px">
  <col style="width: 117px">
  <col style="width: 153px">
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
      <td>A</td>
      <td>Input</td>
      <td>Input A in the formula, an aclTensor on the device.</td>
      <td><ul><li>The data type must form a deduction relationship with C and B.</li>
      <li>The shape (or the shape after transpose) must meet the multiplication condition with B.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>B</td>
      <td>Input</td>
      <td>Input B in the formula, an aclTensor on the device.</td>
      <td><ul><li>The data type must form a deduction relationship with C and A.</li>
      <li>The shape (or the shape after transpose) must satisfy the condition for multiplication with A.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>C</td>
      <td>Input</td>
      <td>Input C in the formula, an aclTensor on the device.</td>
      <td><ul><li>The data type must form a deduction relationship with the result of AB computation.</li>
      <li>The shape can be 1D or 2D and must be consistent with the result of A@B computation or satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a>.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1, 2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Input α in the formula. Floating-point value on the host, representing the coefficient of the product of A and B.</td>
      <td>-</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Input β in the formula, a Host-side floating-point value, indicating the coefficient of C.</td>
      <td>-</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transA</td>
      <td>Input</td>
      <td>Input transA in the formula. Host-side integer. Indicates whether matrix A is transposed. If non-zero, matrix A is transposed and shaped [K,M]; if zero, matrix A is not transposed and shaped [M, K].</td>
      <td>-</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transB</td>
      <td>Input</td>
      <td>Input transB in the formula. Host-side integer. Indicates whether matrix B is transposed. If non-zero, matrix B is transposed and shaped [N, K]; if zero, matrix B is not transposed and shaped [K, N].</td>
      <td>-</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>out in the formula. Device-side aclTensor. The data type must form a derivation relationship with C, and the shape must be consistent with the result of A@B.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Integer on the host, specifying the computation logic used by the Cube unit.</td>
      <td>If the input data types have a deduction relationship, this parameter processes the post-deduction data type by default. Supported enumeration values:<ul>
        <li>0: KEEP_DTYPE, keeps the input data type for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION, allows converting the input data type to lower precision for computation.</li>
        <li>2: USE_FP16, allows converting the input data type to FLOAT16 for computation.</li>
        <li>3: USE_HF32, supports lowering the input precision to data type HFLOAT32 for computation.</li>
        <li>4: USE_FP32_ADD, supports computation in high-precision mode.</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output parameter</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - A supports data types BFLOAT16, FLOAT16, and FLOAT32.
    - B supports data types BFLOAT16, FLOAT16, and FLOAT32.
    - C supports data types BFLOAT16, FLOAT16, and FLOAT32.
    - out supports the following data types: BFLOAT16, FLOAT16, and FLOAT32.
    - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=2: This option does not take effect when the input data type is BFLOAT16 (computation is performed as KEEP_DTYPE), and only a warning is printed.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation. This option is not supported when the input is of other data types.
  <!-- end id6 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>:
    - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=2: This option does not take effect when the input data type is BFLOAT16 (computation is performed as KEEP_DTYPE), and only a warning is printed.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation. This option is not supported when the input is of other data types.
    - cubeMathType=4: when the input data type is FLOAT16/BFLOAT16, the gemm process performs the computation at a higher precision. In this case, broadcasting between input C and the matmul result matrix is currently not supported. When the input data type is FLOAT32 and any of the following conditions is met, the vector core is used for computation to improve result precision (this may degrade operator performance in some use cases). Condition 1: m equals 1 or n equals 1. Condition 2: A is transposed and B is not transposed.

  <!-- end id11 -->
  <!-- npu="910,310p" id7 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>:
    - The data type of A supports FLOAT16 and FLOAT32.
    - The data type of B supports FLOAT16 and FLOAT32.
    - The data type of C supports FLOAT16 and FLOAT32.
    - The data type of out supports FLOAT16 and FLOAT32.
    - BFLOAT16 data type is not supported.
    - When the input data type is FLOAT32, cubeMathType=0 is not supported.
    - For cubeMathType=1, when the input data type is FLOAT32, it is converted to FLOAT16 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=3 is not supported.

  <!-- end id7 -->

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 887px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>A, B, C, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type or data format is out of the supported range.</td>
    </tr>
    <tr>
      <td>A or B is not 2-dimensional, or the shapes do not satisfy the equal k dimension relationship between [m, k] and [k, n] during computation.</td>
    </tr>
    <tr>
      <td>C does not satisfy the broadcast relationship with the result of A@B.</td>
    </tr>
    <tr>
      <td>The shape of out is inconsistent with the shape of the AB computation result.</td>
    </tr>
    <tr>
      <td>cubeMathType is invalid.</td>
    </tr>
  </tbody>
  </table>

## aclnnGemm

- **Parameters**:
  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 230px">
  <col style="width: 150px">
  <col style="width: 750px">
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
      <td>Address of the workspace memory allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the Device side, obtained by the first-phase API aclnnGemmGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>
- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnGemm defaults to a deterministic implementation.

  <!-- end id12 -->

  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: aclnnGemm defaults to a deterministic implementation.

  <!-- end id14 -->

  <!-- npu="910,310p" id8 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>: aclnnGemm defaults to a deterministic implementation.

  <!-- end id8 -->

<!-- npu="950" id13 -->
- <term>Ascend 950PR&950DT products</term>:
  - When the data types of A and B are both FLOAT16 or both BFLOAT16, the computation precision of aclnnGemm is as follows:
    - When the data type of out is FLOAT32, MatMul uses FLOAT32 for accumulation. The intermediate results of beta×C, alpha×(A@B), and the addition of the two parts are all in FLOAT32, and the final result is written to out in FLOAT32.
    - When the data type of out is FLOAT16 or BFLOAT16 and cubeMathType=4, the preceding intermediate results are in FLOAT32, and the final result is converted to the data type specified for out.
    - When the data type of out is FLOAT16 or BFLOAT16 and cubeMathType is not 4, the intermediate results are not converted to FLOAT32.
    - When the shape of C is [n] or [1, n] and alpha=1 and beta=1, C is directly used as the bias input of MatMul, and the addition is performed inside MatMul.

<!-- end id13 -->

<!-- npu="910,310p" id9 -->
- <term>Atlas training products</term>, <term>Atlas inference products</term>: The Cube unit does not support FLOAT32 computation. When the input is FLOAT32, set cubeMathType=1 (ALLOW_FP32_DOWN_PRECISION) to allow the API to internally cast to FLOAT16 for computation.

<!-- end id9 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gemm.h"

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
  // 1. (boilerplate) Initialize the device/stream. For details, see the ACL external API list.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. Customize the construction based on the API.
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

  // Create aclTensor A.
  ret = CreateAclTensor(AHostData, AShape, &ADeviceAddr, aclDataType::ACL_FLOAT, &A);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create aclTensor B.
  ret = CreateAclTensor(BHostData, BShape, &BDeviceAddr, aclDataType::ACL_FLOAT, &B);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create aclTensor C.
  ret = CreateAclTensor(CHostData, CShape, &CDeviceAddr, aclDataType::ACL_FLOAT, &C);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create aclTensor out.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGemm.
  ret = aclnnGemmGetWorkspaceSize(A, B, C, alpha, beta, transA, transB, out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnGemm.
  ret = aclnnGemm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemm failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Wait synchronously for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code according to the specific API definition.
  aclDestroyTensor(A);
  aclDestroyTensor(B);
  aclDestroyTensor(C);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code according to the specific API definition.
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
