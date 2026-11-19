# aclnnMatmul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:17:26.539Z -->

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

- Description: Computes matrix multiplication between tensor self and tensor mat2. (Supports matrix multiplication with 1 to 6 dimensions as input.)
  Similar APIs include aclnnMm (supports 2D tensors as input for matrix multiplication) and aclnnBatchMatmul (supports only 3D matrix multiplication, where the first dimension is the batch).
  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Supports matrix multiplication with 1 to 8 dimensions as input.
  <!-- end id14 -->
- Formula:

  $$
  result=self @ mat2
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnMatmulGetWorkspaceSize" to obtain the workspace size required for the computation and the executor that includes the operator computation flow, and then call "aclnnMatmul" to perform the computation.

```cpp
aclnnStatus aclnnMatmulGetWorkspaceSize(
  const aclTensor *self,
  const aclTensor *mat2,
  aclTensor       *out,
  int8_t           cubeMathType,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMatmul(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnMatmulGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1508px"><colgroup>
  <col style="width: 151px">
  <col style="width: 121px">
  <col style="width: 301px">
  <col style="width: 331px">
  <col style="width: 237px">
  <col style="width: 111px">
  <col style="width: 111px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>description</th>
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
      <td>Represents the first matrix in the matrix multiplication, that is, self in the formula.</td>
      <td><ul><li>The data type needs to be with mat2 to satisfy the data type deduction rules (see also <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li>
      <li>Needs to be with mat2 to satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a>.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-6</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mat2</td>
      <td>Input</td>
      <td>Represents the second matrix in the matrix multiplication, that is, mat2 in the formula.</td>
      <td><ul><li>The data type must satisfy the data type deduction rule with self (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li><li>The Reduce dimension of mat2 must be of the same size as the Reduce dimension of self.</li><li>Must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with self.</li> </ul>
      </td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-6</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Represents the output matrix of matrix multiplication, i.e., out in the formula.</td>
      <td>The data type must satisfy the deduction rule with the data type derived from self and mat2 (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>). For non-empty inputs, the total number of elements in out must be consistent with the result deduced from self and mat2; the shape may differ.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1-6</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Specifies the compute logic of the Cube unit.</td>
      <td>If the input data types have a deduction relationship, this parameter processes the data type after mutual derivation by default. The supported enumeration values are as follows:<ul>
        <li>0: KEEP_DTYPE, keeps the input data type for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION, supports down-precision computation of input data.</li>
        <li>2: USE_FP16, supports down-precision computation of input to FLOAT16.</li>
        <li>3: USE_HF32, supports down-precision computation of input to data type HFLOAT32.</li>
        <li>4: USE_FP32_ADD, supports computation with the high-precision method.</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the Device-side-applied workspace.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output parameter</td>
      <td>Returns the op executor, which includes the computation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas training products</term> and <term>Atlas inference products</term>:
    - BFLOAT16 data type is not supported.
    - When the input data type is FLOAT32, cubeMathType=0 is not supported.
    - For cubeMathType=1, when the input data type is FLOAT32, it is converted to FLOAT16 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=3 is not supported.
    - No processing is performed when cubeMathType is set to 4.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=2: This option is not supported when the input data type is BFLOAT16.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation. This option is not supported when the input is of other data types.
    - cubeMathType=4: When the input data type is FLOAT32 and the k-axis is greater than 2048, group accumulation is used for computation. When the input is of other data types or the k-axis is less than 2048, no processing is performed.
    - Supports matrix multiplication with 1 to 8 dimensions as input.
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:
    - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=2: This option is not supported when the input data type is BFLOAT16.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation. This option is not supported when the input is of other data types.
    - cubeMathType=4, when the input data type is FLOAT32 and the shape is 2D, vector cores are used for computation to improve result precision (this may degrade operator performance in some use cases).

  <!-- end id9 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 809px"><colgroup>
  <col style="width: 257px">
  <col style="width: 121px">
  <col style="width: 431px">
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
      <td>self, mat2, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type and data format of self and mat2 are not in the supported range.</td>
    </tr>
    <tr>
      <td>Data type deduction cannot be performed for self and mat2.</td>
    </tr>
    <tr>
      <td>The deduced data type cannot be converted to the type of the specified output out.</td>
    </tr>
  </tbody>
  </table>

## aclnnMatmul

- **Parameters:**

  <div style="overflow-x: auto;">
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
      <td>Address of the workspace memory applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by the first-phase API aclnnMatmulGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, including the operator compute flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>
  </div>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="910,310p" id10 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>: aclnnMatmul defaults to a deterministic implementation.
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnMatmul defaults to a deterministic implementation.

  <!-- end id11 -->

- Computation consistency description

  <!-- npu="910,310p" id12 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>:
    - When the strong consistency computation feature is enabled, the computation result is deterministic, and multiple executions produce the same output. In addition, the computation result is independent of the data location.
    - aclnnMatmul defaults to a non-consistent implementation. Consistent computation can be enabled through aclrtCtxSetSysParamOpt.
    - For example, during Matrix Multiplication, the accumulation order of different basic blocks may vary, which can cause slight differences in the computation results of the same data across different rows. However, when deterministic computation is enabled, the computation results will be identical as long as the inputs are the same, even across different rows.

  <!-- end id12 -->

<!-- npu="A3,910b" id13 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Data type deduction is supported when the two inputs are BFLOAT16 and FLOAT16 respectively, and when the two inputs are BFLOAT16 and FLOAT32 respectively.
<!-- end id13 -->
- When both self and mat2 are 1-dimensional, cubeMathType does not take effect.

- Shape deduction for one-dimensional inputs:
  - When `self` is one-dimensional `(K)`, it participates in the matrix multiplication with the logical shape `(1, K)` during computation, and the output removes the padded M dimension.
  - When `mat2` is one-dimensional `(K)`, it participates in the matrix multiplication with the logical shape `(K, 1)` during computation, and the output removes the padded N dimension.
  - When both inputs are one-dimensional `(K)`, the output is a scalar; the Reduce dimension K must be equal.
  - For other use cases, first align the batch dimensions except the last two according to the broadcast rules, and then remove the dimensions padded due to one-dimensional inputs according to the preceding rules.
  - For example: `(K) @ (K, N) -> (N)`, `(M, K) @ (K) -> (M)`, `(B, M, K) @ (K, N) -> (B, M, N)`.

## Example

The following is a sample code for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_matmul.h"

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
  // Boilerplate. Initialize resources.
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
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {16, 32};
  std::vector<int64_t> mat2Shape = {32, 16};
  std::vector<int64_t> outShape = {16, 16};
  void* selfDeviceAddr = nullptr;
  void* mat2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* mat2 = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData(512, 1);
  std::vector<float> mat2HostData(512, 1);
  std::vector<float> outHostData(256, 0);
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the mat2 aclTensor.
  ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace the API name with the actual one.
  int8_t cubeMathType = 1;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  // Call the first-phase API of aclnnMatmul.
  ret = aclnnMatmulGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMatmul.
  ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify it based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(mat2);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(mat2DeviceAddr);
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
