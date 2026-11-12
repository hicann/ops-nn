# aclnnTransposeQuantBatchMatMul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:09:40.685Z -->

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

- Description: Performs the quantized matrix multiplication of tensor x1 and tensor x2, and supports the K-C, MX, and T-C [quantization mode](../../../docs/en/context/quant_mode_introduction.md). Only 3D tensors are supported as input. Tensors support transposition, and the transpose sequence changes based on the input value sequence. permX1 represents the transpose sequence of tensor x1, and permX2 represents the transpose sequence of tensor x2. The dimension with a sequence value of 0 is the batch dimension, and matrix multiplication is performed on the other two dimensions.

- Example:
  Assume that the shape of x1 is (M, B, K), the shape of x2 is (B, K, N), x1Scale and x2Scale are not None, and batchSplitFactor equals 1. The shape of the computed output out is (M, B, N).

## Function Prototype

Each operator corresponds to a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnTransposeQuantBatchMatMulGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the process, and then call the "aclnnTransposeQuantBatchMatMul" API to perform the computation.

```cpp
aclnnStatus aclnnTransposeQuantBatchMatMulGetWorkspaceSize(
    const aclTensor   *x1,
    const aclTensor   *x2,
    const aclTensor   *bias,
    const aclTensor   *x1Scale,
    const aclTensor   *x2Scale,
    int32_t            dtype,
    int64_t            groupSize,
    const aclIntArray *permX1,
    const aclIntArray *permX2,
    const aclIntArray *permY,
    int32_t            batchSplitFactor,
    aclTensor         *out,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnTransposeQuantBatchMatMul(
    void               *workspace,
    uint64_t            workspaceSize,
    aclOpExecutor      *executor,
    const aclrtStream   stream)
```

## aclnnTransposeQuantBatchMatMulGetWorkspaceSize

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
        <th>Non-contiguous Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x1 (aclTensor*)</td>
        <td>Input</td>
        <td>Denotes the first matrix of the matrix multiplication.</td>
        <td>          <ul>
            <li>The data type must conform to the data type deduction rule with x2 (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li>
            <li>Currently, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, and HIFLOAT8 are supported. Atlas A2/A3 also support INT8.</li>
          </ul>
        </td>
        <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, HIFLOAT8, INT8</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2 (aclTensor*)</td>
        <td>Input</td>
        <td>Denotes the second matrix of the matrix multiplication.</td>
        <td>        <ul>
            <li>The data type must satisfy the data type deduction rules with x1 (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li>
            <li>The k dimension of x2 must be of the same size as the k dimension of x1.</li>
            <li>Currently, the data types FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, and HIFLOAT8 are supported. Atlas A2/A3 also support INT8.</li>
        </ul>
        </td>
        <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, HIFLOAT8, INT8</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias (aclTensor*)</td>
        <td>Input</td>
        <td>Represents the bias matrix for matmul.</td>
        <td>Reserved parameter. Currently not supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
      <td>x1Scale (aclTensor*)</td>
        <td>Input</td>
        <td>Represents the quantization scale of the left matrix.</td>
        <td>        <ul>
          <li>For the K-C Quantization Scenario, the shape supports one dimension and must equal [m].</li>
          <li>For INT8 W8A8 K-C on Atlas A2/A3, the shape is [M, B].</li>
          <li>For the MX Quantization Scenario, the shape supports four dimensions.</li>
          <li>For the T-C Quantization Scenario, the shape can be empty. If not empty, it supports one dimension and must equal [1].</li>
        </ul>
        </td>
        <td>FLOAT32, FLOAT8_E8M0, UINT64, INT64</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
      <td>x2Scale (aclTensor*)</td>
        <td>Input</td>
        <td>Represents the quantization coefficients of the right matrix.</td>
        <td>        <ul>
          <li>For K-C and T-C Quantization Scenarios, shape is one-dimensional and must be [n].</li>
          <li>For INT8 W8A8 K-C on Atlas A2/A3, the shape is [B, N].</li>
          <li>For the MX Quantization Scenario, shape is four-dimensional.</li>
        </ul>
        </td>
        <td>FLOAT32, FLOAT8_E8M0, UINT64, INT64</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dtype (int32_t)</td>
        <td>Input</td>
        <td>Specifies the data type of the output matrix. Supported values: 1, 27, 34.</td>
        <td>        <ul>
          <li>Value Range 1, indicating the output matrix type is FLOAT16.</li>
          <li>Value Range 27, indicating the output matrix type is BFLOAT16. INT8 W8A8 on Atlas A2/A3 supports only this value.</li>
          <li>Value Range 34, indicating the output matrix type is HIFLOAT8 (only supported in T-C quantization).</li>
        </ul>
        </td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize (int64_t)</td>
        <td>Input</td>
        <td>Specifies the quantization group sizes along the m, n, and k directions.</td>
        <td>Composed of three values: groupSizeM, groupSizeN, and groupSizeK in three directions. Each value occupies 16 bits, taking up the lower 48 bits of the int64_t groupSize (the value of the upper 16 bits in groupSize is invalid).</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permX1 (aclIntArray*)</td>
        <td>Input</td>
        <td>Specifies the transpose sequence of the first matrix in matrix multiplication. It is an aclIntArray on the host.</td>
        <td>Only [1, 0, 2] is supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permX2 (aclIntArray*)</td>
        <td>Input</td>
        <td>Indicates the transpose sequence of the second matrix in the matmul, an aclIntArray on the host.</td>
        <td>        <ul>
          <li>In K-C quantization scenarios, [0, 1, 2] is supported.</li>
          <li>In MX and T-C quantization scenarios, [0, 1, 2] or [0, 2, 1] is supported.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permY (aclIntArray*)</td>
        <td>Input</td>
        <td>Represents the transpose sequence of the matmul output matrix. It is an aclIntArray on the host.</td>
        <td>Only [1, 0, 2] is supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>batchSplitFactor (int32_t)</td>
        <td>Input</td>
        <td>Specifies the size of the B-dimension split in the output matrix of the matrix multiplication. Integer on the host. Currently only 1 is supported.</td>
        <td>Only 1 is supported.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>Represents the output matrix of the matrix multiplication, which is out in the formula.</td>
        <td>Empty tensors are not supported.</td>
        <td>BFLOAT16, FLOAT16, HIFLOAT8</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
            <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td style="white-space: nowrap">executor</td>
        <td>Output</td>
        <td>Returns the op executor that contains the operator computation flow.</td>
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
      <td>x1, x2, out, x1Scale, x2Scale, permX1, permX2, or permY is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type of x1, x2, x1Scale, x2Scale, or out is not in the supported range.</td>
    </tr>
    <tr>
      <td>The shape of x1, x2, x1Scale, x2Scale, or out does not meet the validation conditions.</td>
    </tr>
    <tr>
      <td>The dimension size of x1, x2, permX1, permX2, and permY is not 3.</td>
    </tr>
    <tr>
      <td>batchSplitFactor is not in the supported range.</td>
    </tr>
    <tr>
      <td>The values of permX1, permX2, and permY are not in the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnTransposeQuantBatchMatMul

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the Device, obtained by the first-phase API aclnnTransposeQuantBatchMatMulGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
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

- Deterministic computation description: aclnnTransposeQuantBatchMatMul defaults to a deterministic implementation.

- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Only INT8 W8A8 K-C is supported. x1 and x2 are INT8 ND, x1Scale and x2Scale are FLOAT32, and y is BF16. x1 is [M, B, K], x2 is [B, K, N], x1Scale is [M, B], x2Scale is [B, N], and y is [M, B, N]. permX1 and permY are [1, 0, 2], and permX2 is [0, 1, 2]. K is a multiple of 32, N is a multiple of 16, and batchSplitFactor is 1. On A2, B*K must be less than 65536. dtype supports only 27 (BF16). When x2 is FRACTAL_NZ, use [aclnnTransposeQuantBatchMatMulWeightNz](aclnnTransposeQuantBatchMatMulWeightNz_en.md).

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>:
  - K-C[Quantization Mode](../../../docs/en/context/quant_mode_introduction.md), K supports only 512, and N supports only 128. x1Scale and x2Scale support only 1 dimension. x1Scale requires a shape of (M,), and x2Scale requires a shape of (N,). groupSize supports only the value 0, and other values do not take effect. The x1/x2 inputs support two data types: FLOAT8_E5M2 and FLOAT8_E4M3FN. x1Scale/x2Scale supports only the FLOAT32 data type.
  - MX[Quantization Mode](../../../docs/en/context/quant_mode_introduction.md), which supports two data types: MXFP8 and MXFP4. K supports only multiples of 64. x1Scale and x2Scale support only 4 dimensions. x1Scale requires a shape of (M, B, K/64, 2). When permX2 is [0, 1, 2], x2Scale requires a shape of (B, K/64, N, 2); when permX2 is [0, 2, 1], x2Scale requires a shape of (B, N, K/64, 2). groupSizeM and groupSizeN in groupSize support only 0 or 1, and groupSizeK supports only 32. The x1/x2 inputs support the FLOAT8_E4M3FN and FLOAT4_E2M1 data types. x1Scale/x2Scale supports only the FLOAT8_E8M0 data type.
  - T-C[Quantization Mode](../../../docs/en/context/quant_mode_introduction.md), which supports only static quantization. x1Scale can be set to empty or (1,). x2Scale requires a shape of (N,). The groupSize configuration does not take effect. When x1Scale is not empty, it supports only the INT64/INT64 data type. x2Scale must be preprocessed by [trans_quant_param](../../../quant/trans_quant_param_v2/docs/aclnnTransQuantParamV2.md) and converted to the UINT64/INT64 data type. x1/x2 support only the HIFLOAT8 data type.
  - Constraints on groupSize:
    - Takes effect only in the MX quantization scenario.
    - The input groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK by the following formula. If one or more of them is 0, groupSizeM, groupSizeN, and groupSizeK are reset based on the input shapes of x1/x2/x1Scale/x2Scale for computation. Principle: assume groupSizeM=0. This means that the quantization group value along the M dimension is inferred by the API. The inference formula is groupSizeM = M / scaleM (M must be divisible by scaleM), where M is the same as M in the x1 shape, and scaleM is the same as M in the x1Scale shape.

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

<!-- end id7 -->

## Example

The following sample code is for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include <limits>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_transpose_quant_batch_mat_mul.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
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

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate for resource initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// BF16-to-float conversion function
float bf16_to_float(uint16_t bf16)
{
    uint16_t sign = (bf16 >> 15) & 0x1;
    uint16_t exp = (bf16 >> 7) & 0xFF; // 8-bit exponent
    uint16_t mant = bf16 & 0x7F;

    // Special value handling
    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            // Denormalized BF16 -> float
            return (sign ? -1.0f : 1.0f) * (float)mant * (1.0f / (1 << 7) / std::ldexp(1.0, 126));
        }
    } else if (exp == 255) {
        // Infinity or NaN
        if (mant == 0) {
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        } else {
            return std::numeric_limits<float>::quiet_NaN();
        }
    } else {
        // Normalized numbers
        float f_exp = (float)(exp - 127);      // Offset 127
        float f_mant = (float)mant / (1 << 7); // 7 fractional bits
        float f = (sign ? -1.0f : 1.0f) * (1.0f + f_mant) * (1 << (int)f_exp);
        return f;
    }
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int AclnnTransposeQuantBatchMatmulTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    // Handle the check according to your needs.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs. Customize the construction based on the API.
    int32_t M = 32;
    int32_t K = 512;
    int32_t N = 128;
    int32_t Batch = 16;
    std::vector<int64_t> x1Shape = {M, Batch, K};
    std::vector<int64_t> x2Shape = {Batch, K, N};
    std::vector<int64_t> x1ScaleShape = {M};
    std::vector<int64_t> x2ScaleShape = {N};
    std::vector<int64_t> outShape = {M, Batch, N};
    std::vector<int64_t> permX1Series = {1, 0, 2};
    std::vector<int64_t> permX2Series = {0, 1, 2};
    std::vector<int64_t> permYSeries = {1, 0, 2};
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* x1ScaleDeviceAddr = nullptr;
    void* x2ScaleDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* x1Scale = nullptr;
    aclTensor* x2Scale = nullptr;
    aclTensor* out = nullptr;
    std::vector<int8_t> x1HostData(GetShapeSize(x1Shape), 0x38);
    std::vector<int8_t> x2HostData(GetShapeSize(x2Shape), 0x38);
    std::vector<float> x1ScaleHostData(GetShapeSize(x1ScaleShape), 1);
    std::vector<float> x2ScaleHostData(GetShapeSize(x2ScaleShape), 1);
    std::vector<uint16_t> outHostData(GetShapeSize(outShape), 0); // bf16

    // Create x1 aclTensor.
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1deviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create x2 aclTensor.
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2deviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create x1Scale aclTensor.
    ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1ScaledeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create x2Scale aclTensor.
    ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2ScaledeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray* permX1 = aclCreateIntArray(permX1Series.data(), permX1Series.size());
    aclIntArray* permX2 = aclCreateIntArray(permX2Series.data(), permX2Series.size());
    aclIntArray* permY = aclCreateIntArray(permYSeries.data(), permYSeries.size());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> executorAddrPtr(nullptr, aclrtFree);

    int32_t batchSplitFactor = 1;
    int64_t groupSize = 0;
    int32_t dtype = 27; // bf16

    // Example of calling the aclnnTransposeQuantBatchMatMul API.
    // 3. Call the CANN operator library API. Replace it with the specific API name.
    // Call the first-phase API of aclnnTransposeQuantBatchMatMul.
    ret = aclnnTransposeQuantBatchMatMulGetWorkspaceSize(
        x1, x2, (const aclTensor*)nullptr, x1Scale, x2Scale, dtype, groupSize, permX1, permX2, permY, batchSplitFactor,
        out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeQuantBatchMatMulGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        executorAddrPtr.reset(workspaceAddr);
    }
    // Call the aclnnTransposeQuantBatchMatMul second-phase API.
    ret = aclnnTransposeQuantBatchMatMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeQuantBatchMatMul failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from Device memory to Host memory. Modify the code based on the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0); // bf16
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    float resultDataBF16 = 0;
    for (int64_t i = 0; i < size; i++) {
        resultDataBF16 = bf16_to_float(resultData[i]);
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataBF16);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = AclnnTransposeQuantBatchMatmulTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeQuantBatchMatMulTest failed. ERROR: %d\n", ret);
                   return ret);
    Finalize(deviceId, stream);
    return 0;
}
```
