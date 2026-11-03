# aclnnTransposeBatchMatMulWeightNz

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:08:58.892Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Not Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not Supported
<!-- end id6 -->

## Function

- Description: Computes matrix multiplication of tensor x1 and tensor x2. x1 must be in ND format, and x2 must be in NZ format. x1 must be 3-dimensional, and x2 must be 5-dimensional. Tensor transposition is supported. The transpose sequence varies based on the input sequence. permX1 denotes the transpose sequence of tensor x1, and permX2 denotes the transpose sequence of tensor x2. The dimension with sequence value 0 is the batch dimension, and the other two dimensions are used for matrix multiplication.

- Example: (viewshape perspective corresponding to the conversion of x2 from NZ to ND)
  - When the shape of x1 is (B, M, K), the shape of x2 is (B, K, N), scale is None, and batchSplitFactor equals 1, the shape of the computed output out is (M, B, N).
  - When the shape of x1 is (B, M, K), the shape of x2 is (B, K, N), scale is not None, and batchSplitFactor equals 1, the shape of the computed output out is (M, 1, B * N).
  - When the shape of x1 is (B, M, K), the shape of x2 is (B, K, N), scale is None, and batchSplitFactor is greater than 1, the shape of the compute output out is (batchSplitFactor, M, B * N / batchSplitFactor).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnTransposeBatchMatMulWeightNzGetWorkspaceSize" API to obtain input parameters and calculate the required workspace size based on the workflow. Then call the "aclnnTransposeBatchMatMulWeightNz" API to perform computation.

```cpp
aclnnStatus aclnnTransposeBatchMatMulWeightNzGetWorkspaceSize(
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
aclnnStatus aclnnTransposeBatchMatMulWeightNz(
    void               *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor      *executor,
    const aclrtStream  stream)
```

## aclnnTransposeBatchMatMulWeightNzGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed;width: 1567px"><colgroup>
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
        <th style="white-space: nowrap">Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th style="white-space: nowrap">Dimension</th>
        <th>Non-Contiguous Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x1</td>
        <td>Input</td>
        <td>Represents the first matrix in matrix multiplication.</td>
        <td>          <ul>
            <li>The data type needs to be consistent with x2 to satisfy the data type derivation rule (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li>
            <li>Data type derivation is not supported when inputs x1 and x2 are BFLOAT16 and FLOAT16 respectively.</li>
            <li>Data type derivation is not supported when inputs x1 and x2 are BFLOAT16 and FLOAT32 respectively.</li>
          </ul>
        </td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>Input</td>
        <td>Represents the second matrix in matrix multiplication.</td>
        <td>        <ul>
            <li>The data type needs to be consistent with x1 according to the data type derivation rule (See <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li>
            <li>The size of the Reduce dimension of x2 needs to be equal to that of x1.</li>
            <li>Data type derivation is not supported when input x1 and x2 are BFLOAT16 and FLOAT16 respectively.</li>
            <li>The dimensions of the NZ format are (b, n1, k1, k0, n0), where k0 = 16 and n0 = 16. The k in the x1 shape and k1 in the x2 shape need to satisfy the following relationship: ceil(k, k0) = k1. The n1 in the x2 shape and n in out satisfy the following relationship: ceil(n, n0) = n1.</li>
        </ul>
        </td>
        <td>BFLOAT16, FLOAT16</td>
        <td>NZ</td>
        <td>5</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>Input</td>
        <td>Represents the bias matrix of matrix multiplication.</td>
        <td>        <ul>
            <li>Reserved parameter, currently not supported.</li>
        </ul>
        </td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
      <td>scale</td>
        <td>Input (optional)</td>
        <td>Represents the quantization coefficient of the output matrix. This can be enabled when the input is FLOAT16 and the output is INT8.</td>
        <td>        <ul>
            <li>Only one-dimensional shape is supported, and it must be equal to [b*n].</li>
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
        <td>Represents the transpose sequence of the first matrix in matrix multiplication.</td>
        <td>        <ul>
          <li> Supports [0, 1, 2] and [1, 0, 2].</li>
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
        <td>Represents the transpose sequence of the second matrix in the matrix multiplication.</td>
        <td>        <ul>
          <li> Supports [0, 1, 2] and [0, 2, 1].</li>
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
        <td>Represents the transpose sequence of the output matrix in the matrix multiplication.</td>
        <td>        <ul>
            <li>Supports [1, 0, 2].</li>
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
        <td>Specifies the compute logic of the Cube unit.</td>
        <td>If the input data types have a deduction relationship, this parameter processes the derived data type by default. The enumeration values are as follows:<ul>
            <li>0: KEEP_DTYPE, computes with the input data type unchanged.</li>
            <li>1: ALLOW_FP32_DOWN_PRECISION, supports converting input data to a lower precision for computation. When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. For other data types, the input type is kept for computation.</li>
            <li>2: USE_FP16, supports converting the input to FLOAT16 for computation. This option does not take effect when the input data type is BFLOAT16; computation is performed with the input data type.</li>
            <li>3: USE_HF32, supports converting the input to data type HFLOAT32 for computation. When the input data type is FLOAT32, it is converted to HFLOAT32 for computation.</li></ul>
        </td>
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>batchSplitFactor</td>
        <td>Input</td>
        <td>Specifies the split size of dimension B in the output matrix of matrix multiplication.</td>
        <td>        <ul>
          <li>Value range: [1, B]. The value must be divisible by B.</li>
          <li>When scale is not empty, batchSplitFactor can only be equal to 1.</li>
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
        <td>Represents the output matrix of matmul, i.e., out in the formula.</td>
        <td>        <ul>
          <li> The data type needs to be consistent with the data type derived from x1 and x2 (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li>
          <li> When scale has a value, the output shape is (M, 1, B * N).</li>
        </ul>
        <ul>
          <li>When batchSplitFactor is greater than 1, the output shape of out is (batchSplitFactor, M, B * N / batchSplitFactor).</li>
          <ul>
            <li> Example 1: M, K, N, B = 32, 512, 128, 16; when batchSplitFactor = 2, the output shape size of out is (2, 32, 1024).</li>
            <li> Example 2: M, K, N, B = 32, 512, 128, 16; when batchSplitFactor = 4, the output shape size of out is (4, 32, 512).</li>
          </ul>
        </ul>
        </td>
        <td>BFLOAT16, FLOAT16, INT8</td>
        <td>ND</td>
        <td>3</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be applied for on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td style="white-space: nowrap">executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the compute process of the operator.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      </tbody>
      </table>

- **Return value**

  aclnnStatus: return status code, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

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
      <td>The input x1, x2, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="16">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="16">161002</td>
      <td>The data type of x1, x2, or out is not in the supported range, or the data types of x1 and x2 are different.</td>
    </tr>
    <tr>
      <td>The dimension size of the ViewShape of x1 or x2 is not equal to 3.</td>
    </tr>
    <tr>
      <td>The length of permX1, permX2, or permY is not equal to 3, or their values are not in the supported range.</td>
    </tr>
    <tr>
      <td>The second dimension of x1 is not equal to the first dimension of x2, or the batch dimensions of x1 and x2 are not equal.</td>
    </tr>
    <tr>
      <td>x1 or x2 is an empty tensor.</td>
    </tr>
    <tr>
      <td>The StorageFormat of x2 is not FORMAT_FRACTAL_NZ.</td>
    </tr>
    <tr>
      <td>The StorageFormat of x1 or out is FORMAT_FRACTAL_NZ.</td>
    </tr>
    <tr>
      <td>The StorageShape dimension size of x2 is not equal to 5, or the k-axis or n-axis size of x2 is 1.</td>
    </tr>
    <tr>
      <td>The data type of scale is not in the supported range, or its shape is not one-dimensional, or its length is not equal to B*N.</td>
    </tr>
    <tr>
      <td>bias is not null.</td>
    </tr>
    <tr>
      <td>The value of cubeMathType is not in the supported range, or the current product does not support the configured data type combination.</td>
    </tr>
    <tr>
      <td>batchSplitFactor is less than or equal to 0, or cannot evenly divide batch when it is greater than 1, or is not equal to 1 when scale is not null.</td>
    </tr>
    <tr>
      <td>B, N, K, or B*K is not in the supported range.</td>
    </tr>
    <tr>
      <td>When x2 is transposed, the combination of permX2, scale, or batchSplitFactor is not in the supported range.</td>
    </tr>
    <tr>
      <td>batch*N is greater than INT64_MAX, or when scale is not null, B*N is not in the supported range.</td>
    </tr>
    <tr>
      <td>The shape of out is inconsistent with the calculated shape.</td>
    </tr>
  </tbody>
  </table>

## aclnnTransposeBatchMatMulWeightNz

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
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
    <td>Memory address of the workspace on the Device side.</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace on the Device side, obtained by the first-phase API aclnnTransposeBatchMatMulWeightNzGetWorkspaceSize.</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>Input</td>
    <td>Op executor, including the operator compute flow.</td>
  </tr>
  <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Stream for task execution.</td>
  </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnTransposeBatchMatMulWeightNz defaults to a deterministic implementation.

<!-- npu="A3,910b" id7 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - The value range of B is [1, 65536), and the value range of N is [1, 65536).
  - When the input shape of x1 is (B, M, K), K <= 65535; when the input shape of x1 is (M, B, K), if scale is not empty or batchSplitFactor is not equal to 1, both K and B * K must be less than 65536.
  - In the ND format corresponding to the NZ format of x2, the second and third dimensions must be divisible by 16.
  - The StorageFormat of x2 must be NZ format.
  - x1 and x2 must have the same dtype.
  - permX2 only supports input [0, 1, 2].
  - When scale is not empty, the product of B and N must be less than 65536, and only the type derivation with input FLOAT16 and output INT8 is supported.

<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>:
  - permX2 supports input [0, 1, 2] and [0, 2, 1].
  - When scale is not empty, batchSplitFactor must equal 1, and only type inference with FLOAT16 input and INT8 output is supported.

<!-- end id8 -->
- self only supports 3 dimensions, and mat2 only supports the Ascend private format. Before calling this API, you must complete the conversion of mat2 from ND to the Ascend private format.
- The case where one of the last two axes of mat2 is 1, i.e., k=1 or n=1, is not supported.

## Example

<!-- npu="A3,910b" id9 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  The following is the sample code for the use case where the data type of x1 and x2 is float16 and x2 is in NZ format, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include <cmath>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_transpose_batch_mat_mul.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  // Convert the uint16_t representation of FP16 to the float representation
  float Fp16ToFloat(uint16_t h)
  {
      int s = (h >> 15) & 0x1;  // sign
      int e = (h >> 10) & 0x1F; // exponent
      int f = h & 0x3FF;        // fraction
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

  int Init(int32_t deviceId, aclrtStream* stream)
  {
      // Boilerplate: resource initialization.
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate memory on the Device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      // Call aclrtMemcpy to copy data on the Host to the Device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensorWeight(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray* mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, aclDataType::ACL_FLOAT16, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return ret);
      size *= sizeof(T);

      // Call aclrtMalloc to allocate memory on the device.
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy data from host to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, storageShape.data(),
          storageShape.size(), *deviceAddr);
      return 0;
  }

  int main()
  {
      // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
      // Set deviceId according to the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      // Handle the check as needed.
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct input and output. Custom construction is required based on the API.
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
      void* outDeviceAddr = nullptr;
      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* out = nullptr;
      // Use uint16_t to store float16 data.
      std::vector<uint16_t> x1HostData(GetShapeSize(x1Shape), 0x3C00); // 1.0 of float16
      std::vector<uint16_t> x2HostData(GetShapeSize(x2Shape), 0x3C00); // 1.0 of float16
      std::vector<uint16_t> outHostData(GetShapeSize(outShape), 0);
      int8_t cubeMathType = 0; // Consistent with BMMV3
      int32_t batchSplitFactor = 1;

      // Create x1 aclTensor (ND format).
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create x2 aclTensor (for NZ format).
      ret = CreateAclTensorWeight(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create out aclTensor (ND format).
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      aclIntArray* permX1 = aclCreateIntArray(permX1Series.data(), permX1Series.size());
      aclIntArray* permX2 = aclCreateIntArray(permX2Series.data(), permX2Series.size());
      aclIntArray* permY = aclCreateIntArray(permYSeries.data(), permYSeries.size());

      // 3. Call TransWeight first to convert x2 to NZ format.
      uint64_t transWorkspaceSize = 0;
      aclOpExecutor* transExecutor = nullptr;
      ret = aclnnTransMatmulWeightGetWorkspaceSize(x2, &transWorkspaceSize, &transExecutor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      // Allocate device memory based on the workspaceSize computed by the first-phase API.
      void* transWorkspaceAddr = nullptr;
      if (transWorkspaceSize > 0) {
          ret = aclrtMalloc(&transWorkspaceAddr, transWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate trans workspace failed. ERROR: %d\n", ret); return ret);
      }

      // Call the second-phase API aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeight(transWorkspaceAddr, transWorkspaceSize, transExecutor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // 4. Call the dedicated TBMM Weight NZ API.
      uint64_t tbmmWorkspaceSize = 0;
      aclOpExecutor* tbmmExecutor = nullptr;
      ret = aclnnTransposeBatchMatMulWeightNzGetWorkspaceSize(
          x1, x2, (const aclTensor*)nullptr, (const aclTensor*)nullptr, permX1, permX2, permY, cubeMathType,
          batchSplitFactor, out, &tbmmWorkspaceSize, &tbmmExecutor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeBatchMatMulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      // Allocate device memory based on the workspaceSize computed by the first-phase API.
      void* tbmmWorkspaceAddr = nullptr;
      if (tbmmWorkspaceSize > 0) {
          ret = aclrtMalloc(&tbmmWorkspaceAddr, tbmmWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate TBMM workspace failed. ERROR: %d\n", ret); return ret);
      }

      // Call the second-phase API aclnnTransposeBatchMatMulWeightNz.
      ret = aclnnTransposeBatchMatMulWeightNz(tbmmWorkspaceAddr, tbmmWorkspaceSize, tbmmExecutor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeBatchMatMulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 5. (boilerplate) Synchronize and wait for task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 6. Obtain the output value and copy the result from the device memory to the host side.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

      // Print the result (convert uint16_t to float for display).
      int64_t max_print_size = 8;
      for (int64_t i = 0; i < max_print_size; i++) {
          float fp16Float = Fp16ToFloat(resultData[i]);
          LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
      }

      // 7. Release aclTensor and resources.
      aclDestroyTensor(x1);
      aclDestroyTensor(x2);
      aclDestroyTensor(out);
      aclDestroyIntArray(permX1);
      aclDestroyIntArray(permX2);
      aclDestroyIntArray(permY);

      // 8. Release device resources.
      aclrtFree(x1DeviceAddr);
      aclrtFree(x2DeviceAddr);
      aclrtFree(outDeviceAddr);

      if (transWorkspaceSize > 0) {
          aclrtFree(transWorkspaceAddr);
      }
      if (tbmmWorkspaceSize > 0) {
          aclrtFree(tbmmWorkspaceAddr);
      }

      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```

<!-- end id9 -->
<!-- npu="950" id10 -->
- <term>Ascend 950PR&950DT products</term>:
  The following is the sample code for the use case where the data types of x1 and x2 are float16 and x2 is in NZ format. For reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <vector>
  #include <cmath>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_transpose_batch_mat_mul.h"
  #include "aclnnop/aclnn_npu_format_cast.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  // Convert the uint16_t representation of FP16 to the float representation.
  float Fp16ToFloat(uint16_t h)
  {
      int s = (h >> 15) & 0x1;  // sign
      int e = (h >> 10) & 0x1F; // exponent
      int f = h & 0x3FF;        // fraction
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

  int Init(int32_t deviceId, aclrtStream* stream)
  {
      // Boilerplate: resource initialization.
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate Device-side memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      // Call aclrtMemcpy to copy Host-side data to Device-side memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensorWithFormat(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t** storageShape,
      uint64_t* storageShapeSize, void** deviceAddr, aclDataType dataType, aclTensor** tensor, aclFormat format)
  {
      auto size = hostData.size() * sizeof(T);
      // Call aclrtMalloc to allocate device-side memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device-side memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, format, *storageShape, *storageShapeSize, *deviceAddr);
      return 0;
  }

  int main()
  {
      // 1. (Boilerplate) Initialize device/stream. See the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      // Handle the check as needed.
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the inputs and outputs. Use custom construction based on the API.
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
      void* outDeviceAddr = nullptr;
      void* x2NzDeviceAddr = nullptr;
      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* out = nullptr;
      aclTensor* x2NZ = nullptr;

      // Use uint16_t to store float16 data.
      std::vector<uint16_t> x1HostData(GetShapeSize(x1Shape), 0x3C00); // 1.0 in float16.
      std::vector<uint16_t> x2HostData(GetShapeSize(x2Shape), 0x3C00); // 1.0 in float16.
      std::vector<uint16_t> outHostData(GetShapeSize(outShape), 0);
      int8_t cubeMathType = 0; // Consistent with BMMV3.
      int32_t batchSplitFactor = 1;

      // Create x1 aclTensor (ND format).
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create x2 aclTensor (ND format).
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create out aclTensor (ND format).
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      aclIntArray* permX1 = aclCreateIntArray(permX1Series.data(), permX1Series.size());
      aclIntArray* permX2 = aclCreateIntArray(permX2Series.data(), permX2Series.size());
      aclIntArray* permY = aclCreateIntArray(permYSeries.data(), permYSeries.size());

      // 3. Convert the weight tensor from ND to NZ. Call the npu_foramt_cast API.
      aclDataType additionalDtype = aclDataType::ACL_FLOAT16;
      aclDataType srcDtype = aclDataType::ACL_FLOAT16;
      int64_t* dstShape = nullptr;
      uint64_t dstShapeSize = 0;
      int actualFormat;
      aclOpExecutor* executor = nullptr;

      uint64_t formatCastWorkspaceSize = 0;
      uint64_t tbmmWorkspaceSize = 0;
      void* formatCastWorkspaceAddr = nullptr;
      void* tbmmWorkspaceAddr = nullptr;
      // Compute the shape and format of the target tensor.
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(x2, 29, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
      CHECK_RET(
          ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret); return ret);

      ret = CreateAclTensorWithFormat(
          x2HostData, x2Shape, &dstShape, &dstShapeSize, &x2NzDeviceAddr, srcDtype, &x2NZ,
          static_cast<aclFormat>(actualFormat));
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API aclnnNpuFormatCastGetWorkspaceSize.
      ret = aclnnNpuFormatCastGetWorkspaceSize(x2, x2NZ, &formatCastWorkspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      // Allocate device memory based on the workspaceSize computed by the first-phase API.
      if (formatCastWorkspaceSize > 0) {
          ret = aclrtMalloc(&formatCastWorkspaceAddr, formatCastWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate format cast workspace failed. ERROR: %d\n", ret); return ret);
      }

      // Call the second-phase API aclnnNpuFormatCast.
      ret = aclnnNpuFormatCast(formatCastWorkspaceAddr, formatCastWorkspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

      // 4. Synchronously wait for the format conversion task to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Call the dedicated TBMM Weight NZ API.
      ret = aclnnTransposeBatchMatMulWeightNzGetWorkspaceSize(
          x1, x2NZ, (const aclTensor*)nullptr, (const aclTensor*)nullptr, permX1, permX2, permY, cubeMathType,
          batchSplitFactor, out, &tbmmWorkspaceSize, &executor);
      CHECK_RET(
          ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeBatchMatMulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
          return ret);

      // Allocate device memory based on the workspaceSize computed by the first-phase API.
      if (tbmmWorkspaceSize > 0) {
          ret = aclrtMalloc(&tbmmWorkspaceAddr, tbmmWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate TBMM workspace failed. ERROR: %d\n", ret); return ret);
      }

      // Call the second-phase API aclnnTransposeBatchMatMulWeightNz.
      ret = aclnnTransposeBatchMatMulWeightNz(tbmmWorkspaceAddr, tbmmWorkspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeBatchMatMulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 6. (boilerplate) Synchronously wait for the task to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 7. Obtain the output value and copy the result on device memory to the Host.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

      // Print the result.
      int64_t max_print_size = 8;
      for (int64_t i = 0; i < max_print_size; i++) {
          float fp16Float = Fp16ToFloat(resultData[i]);
          LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
      }

      // 8. Release the aclTensor and resources.
      aclDestroyTensor(x1);
      aclDestroyTensor(x2);
      aclDestroyTensor(x2NZ);
      aclDestroyTensor(out);
      aclDestroyIntArray(permX1);
      aclDestroyIntArray(permX2);
      aclDestroyIntArray(permY);

      // 9. Release device resources.
      aclrtFree(x1DeviceAddr);
      aclrtFree(x2DeviceAddr);
      aclrtFree(x2NzDeviceAddr);
      aclrtFree(outDeviceAddr);
      if (formatCastWorkspaceSize > 0) {
          aclrtFree(formatCastWorkspaceAddr);
      }
      if (tbmmWorkspaceSize > 0) {
          aclrtFree(tbmmWorkspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```

<!-- end id10 -->
