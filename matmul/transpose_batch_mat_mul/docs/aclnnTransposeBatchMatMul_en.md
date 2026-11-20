# aclnnTransposeBatchMatMul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:04:46.896Z -->

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

- Description: Computes the matrix multiplication of tensor x1 and tensor x2. Only 3D tensors are supported. Tensor transposition is supported. The transpose sequence changes according to the input array. permX1 and permX2 give the input-view axis positions of logical [B,M,K] and [B,K,N], respectively. Their values are axis indices, not dimension sizes.

- Example:
  - When the shape of x1 is (B, M, K), the shape of x2 is (B, K, N), scale is None, and batchSplitFactor equals 1, the shape of the computed output out is (M, B, N).
  - When the shape of x1 is (B, M, K), the shape of x2 is (B, K, N), scale is not None, and batchSplitFactor equals 1, the shape of the computed output out is (M, 1, B * N).
  - When the shape of x1 is (B, M, K), the shape of x2 is (B, K, N), scale is None, and batchSplitFactor is greater than 1, the shape of the compute output out is (batchSplitFactor, M, B * N / batchSplitFactor).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnTransposeBatchMatMulGetWorkspaceSize" first to obtain the input parameters and compute the required workspace size according to the process. Then call "aclnnTransposeBatchMatMul" to perform the computation.

```cpp
aclnnStatus aclnnTransposeBatchMatMulGetWorkspaceSize(
    const aclTensor    *x1,
    const aclTensor    *x2,
    const aclTensor    *bias,
    const aclTensor    *scale,
    const aclIntArray  *permX1,
    const aclIntArray  *permX2,
    const aclIntArray  *permY,
    int8_t             cubeMathType,
    const int32_t      batchSplitFactor,
    aclTensor          *out,
    uint64_t           *workspaceSize,
    aclOpExecutor      **executor)
```

```cpp
aclnnStatus aclnnTransposeBatchMatMul(
    void               *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor      *executor,
    const aclrtStream  stream)
```

## aclnnTransposeBatchMatMulGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed;width: 1545px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 350px">
    <col style="width: 210px">
    <col style="width: 120px">
    <col style="width: 130px">
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
        <th>Non-Contiguous Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x1</td>
        <td>Input</td>
        <td>Represents the first matrix of the matrix multiplication.</td>
        <td>          <ul>
            <li>The data type must satisfy the data type derivation rule with x2 (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li>
            <li>Supported data types: BFLOAT16, FLOAT16, and FLOAT32.</li>
            <li>Type derivation is not supported when x1 and x2 are BFLOAT16 and FLOAT16 respectively.</li>
            <li>Type derivation is not supported when x1 and x2 are BFLOAT16 and FLOAT32 respectively.</li>
          </ul>
        </td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>Input</td>
        <td>Represents the second matrix of the matrix multiplication.</td>
        <td>        <ul>
            <li>The data type needs to comply with the type derivation rules with x1 (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li>
            <li>The Reduce dimension of x2 needs to be the same size as the Reduce dimension of x1.</li>
            <li>Supported data types are BFLOAT16, FLOAT16, and FLOAT32.</li>
            <li>Type derivation is not supported when inputs x1 and x2 are BFLOAT16 and FLOAT16 respectively.</li>
            <li>Type derivation is not supported when inputs x1 and x2 are BFLOAT16 and FLOAT32 respectively.</li>
        </ul>
        </td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>Optional input</td>
        <td>Represents the bias matrix of matrix multiplication.</td>
        <td>        <ul>
            <li>Pass a null pointer when unused. For product-specific support, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1 / 2</td>
        <td>-</td>
      </tr>
      <tr>
      <td>scale</td>
        <td>Optional input</td>
        <td>Represents the quantization coefficient of the output matrix. It can be enabled when the input is FLOAT16 and the output is INT8. Use the aclnnTransQuantParamV2 API to complete the conversion from FLOAT32 to UINT64/INT64.</td>
        <td>        <ul>
            <li>Only one-dimensional shapes are supported, and the shape must be equal to [b*n].</li>
        </ul>
        </td>
        <td>INT64, UINT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>permX1</td>
        <td>Input</td>
        <td>Represents the transpose sequence of the first matrix in the matrix multiplication, an aclIntArray on the host side.</td>
        <td>        <ul>
          <li>Supports [0, 1, 2], [1, 0, 2], and [0, 2, 1]. For product-specific support, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permX2</td>
        <td>Input</td>
        <td>Represents the transpose sequence of the second matrix in the matrix multiplication, an aclIntArray on the host side.</td>
        <td>        <ul>
          <li>Supports [0, 1, 2], [0, 2, 1], [1, 0, 2], and [1, 2, 0]. For product-specific support, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permY</td>
        <td>Input</td>
        <td>Represents the transpose sequence of the output matrix of the matrix multiplication, an aclIntArray on the host side.</td>
        <td>        <ul>
            <li>Supports [1, 0, 2] and [0, 1, 2]. For product-specific support, see <a href="#constraints">Constraints</a>.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>cubeMathType</td>
        <td>Input</td>
        <td>Specifies the compute logic of the Cube unit. Host-side integer.</td>
        <td>If the input data types have a deduction relationship, this parameter is applied to the deduced data type by default. The enumeration values are as follows:<ul>
            <li>0: KEEP_DTYPE, which keeps the input data type for computation.</li>
            <li>1: ALLOW_FP32_DOWN_PRECISION, which supports computing the input data at a reduced precision. When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the data is of another data type, the input type is kept for computation.</li>
            <li>2: USE_FP16, which supports downgrading the input to FLOAT16 for computation. This option is not supported when the input data type is BFLOAT16.</li>
            <li>3: USE_HF32, which supports downgrading the input precision to data type HFLOAT32 for computation. When the input data type is FLOAT32, it is converted to HFLOAT32 for computation.</li></ul>
        </td>
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>batchSplitFactor</td>
        <td>Input</td>
        <td>Specifies the split size along dimension B in the output matrix of the matrix multiplication. Host-side integer.</td>
        <td>        <ul>
          <li>Value range: [1, B]. The value must be divisible by B.</li>
          <li>When scale is not empty, batchSplitFactor can only equal 1.</li>
        </ul>
        </td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Represents the output matrix of matrix multiplication, which is out in the formula.</td>
        <td>        <ul>
          <li> The data type must be consistent with the data type of x1 and x2 after deduction (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li>
          <li> When scale is not empty, the output shape is (M, 1, B * N).</li>
        </ul>
        <ul>
          <li>When batchSplitFactor is greater than 1, the output shape of out is (batchSplitFactor, M, B * N / batchSplitFactor).</li>
          <ul>
            <li> Example 1: M, K, N, B = 32, 512, 128, 16; when batchSplitFactor = 2, the size of the output shape of out is (2, 32, 1024).</li>
            <li> Example 2: M, K, N, B = 32, 512, 128, 16; when batchSplitFactor = 4, the size of the output shape of out is (4, 32, 512).</li>
          </ul>
        </ul>
        </td>
        <td>BFLOAT16, FLOAT16, FLOAT32, INT8</td>
        <td>ND</td>
        <td>3</td>
        <td>-</td>
      </tr>
      </tbody>
      </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x1, x2, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of x1, x2, or out is not in the supported range.</td>
    </tr>
    <tr>
      <td>The second dimension of x1 is not equal to the first dimension of x2.</td>
    </tr>
    <tr>
      <td>The dimension size of x1 or x2 is not equal to 3.</td>
    </tr>
    <tr>
      <td>The data type of scale is not within the supported range.</td>
    </tr>
    <tr>
      <td>The value of batchSplitFactor is not within the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnTransposeBatchMatMul

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1143px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 825px">
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
      <td>Workspace size allocated on the Device side, obtained through the first-phase API aclnnTransposeBatchMatMulGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, including the operator compute flow.</td>
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

- Deterministic computation description: aclnnTransposeBatchMatMul defaults to a deterministic implementation.

<!-- npu="A3,910b" id7 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - The value range of B is [1, 65536), and the value range of N is [1, 65536).
    - When the input shape of x1 is (B, M, K), K <= 65535 is required. When the input shape of x1 is (M, B, K) and B * K > 65535, scale is not supported, batchSplitFactor can only equal 1, and permX1 must be [1, 0, 2].
    - When the input of permX2 is [0, 2, 1], scale is not supported, batchSplitFactor can only equal 1, and permX1 must be [1, 0, 2].
    - When scale is not empty, batchSplitFactor can only equal 1, the product of B and N must be less than 65536, and only the type derivation with FLOAT16 input and INT8 output is supported.
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>:
    - permX1 supports [0, 1, 2], [1, 0, 2], and [0, 2, 1].
    - permX2 supports [0, 1, 2], [0, 2, 1], [1, 0, 2] and [1, 2, 0].
    - permY supports [1, 0, 2] and [0, 1, 2]. [0, 1, 2] produces the batch-first layout (B, M, N) and requires no scale and batchSplitFactor=1.
    - When scale is not empty, batchSplitFactor must equal 1, and only type inference with FLOAT16 input and INT8 output is supported.
    - bias is supported with shape (N,) or (1,N) and the same data type as x1 or FLOAT32. It is shared across batches and rows and requires scale=None and batchSplitFactor=1.
    - permX1=[0, 2, 1] or permX2=[1, 0, 2]/[1, 2, 0] requires permY=[0, 1, 2], no scale and batchSplitFactor=1.
    - ND NBK B uses a contiguous (N, B, K) tensor and permX2=[1, 2, 0].
<!-- end id8 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_transpose_batch_mat_mul.h"

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

// Convert the uint16_t representation of FP16 to the float representation.
float Fp16ToFloat(uint16_t h) {
  int s = (h >> 15) & 0x1;              // sign
  int e = (h >> 10) & 0x1F;             // exponent
  int f =  h        & 0x3FF;            // fraction
  if (e == 0) {
    // Zero or Denormal
    if (f == 0) {
      return s ? -0.0f : 0.0f;
    }
    // Denormals
    float sig = f / 1024.0f;
    float result = sig * pow(2, -24);
    return s ? -result : result;
  } else if (e == 31) {
      // Infinity or NaN
      return f == 0 ? (s ? -INFINITY : INFINITY) : NAN;
  }
  // Normalized
  float result = (1.0f + f / 1024.0f) * pow(2, e - 15);
  return s ? -result : result;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate, resource initialization.
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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to device memory.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Custom construction is required based on the API.
  int32_t M = 32;
  int32_t K = 512;
  int32_t N = 128;
  int32_t Batch = 16;
  std::vector<int64_t> x1Shape = {M, Batch, K};
  std::vector<int64_t> x2Shape = {Batch, K, N};
  std::vector<int64_t> outShape = {M, Batch, N};
  std::vector<int64_t> permX1Series = {1, 0, 2};
  std::vector<int64_t> permX2Series = {0, 1, 2};
  std::vector<int64_t> permYSeries = {1, 0, 2};
  void* x1DeviceAddr = nullptr;
  void* x2DeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* x1 = nullptr;
  aclTensor* x2 = nullptr;
  aclTensor* scale = nullptr;
  aclTensor* out = nullptr;
  std::vector<uint16_t> x1HostData(GetShapeSize(x1Shape),0x3C00);
  std::vector<uint16_t> x2HostData(GetShapeSize(x2Shape),0x3C00);
  std::vector<uint16_t> outHostData(GetShapeSize(outShape),0);
  int8_t cubeMathType = 1;
  int32_t batchSplitFactor = 1;

  // Create x1 aclTensor.
  ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create x2 aclTensor.
  ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  aclIntArray *permX1 = aclCreateIntArray(permX1Series.data(), permX1Series.size());
  aclIntArray *permX2 = aclCreateIntArray(permX2Series.data(), permX2Series.size());
  aclIntArray *permY = aclCreateIntArray(permYSeries.data(), permYSeries.size());
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // Example of the aclnnTransposeBatchMatMul API call.
  // 3. Call the CANN operator library API. Change it to the specific API name.
  // Call the aclnnTransposeBatchMatMul first-phase API.
  ret = aclnnTransposeBatchMatMulGetWorkspaceSize(x1, x2, (const aclTensor*)nullptr, (const aclTensor*)nullptr,
                                                  permX1, permX2, permY, cubeMathType, batchSplitFactor, out,
                                                  &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeBatchMatMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnTransposeBatchMatMul second-phase API.
  ret = aclnnTransposeBatchMatMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeBatchMatMul failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from Device memory to Host memory. Modify the code based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    float fp16Float = Fp16ToFloat(resultData[i]);
    LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(x1);
  aclDestroyTensor(x2);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the specific API definition.
  aclrtFree(x1DeviceAddr);
  aclrtFree(x2DeviceAddr);
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
