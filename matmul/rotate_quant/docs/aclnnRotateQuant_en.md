# aclnnRotateQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:58:38.607Z -->

## Applicable Products

<!-- npu="950" id1 -->
- Ascend 950PR&950DT products: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 products: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- Atlas 200I/500 A2 inference products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- Atlas training products: Not supported
<!-- end id6 -->

## Function

- Description: Performs rotation transformation on tensor x, then applies an optional clamp operation, and finally performs symmetric dynamic quantization (output data type: int8 or quint4x2) or MX quantization (output data type: FLOAT4 or FLOAT8 types).
- Formula:
  1. Rotation transformation

    $$
    Y = (x.\text{reshape}(*,k) @ \text{rotation}).\text{reshape}(m, n)
    $$

    Where: $\mathbf{x} \in \mathbb{R}^{m \times n}$, $\mathbf{Y} \in \mathbb{R}^{m \times n}$, $\mathbf{rotation} \in \mathbb{R}^{k \times k}$.

  2. When alpha is within the valid range (0.0, 1.0), perform the clamp computation

    $$
    groupMaxVal = GroupMax(|Y|) \\
    limit = alpha * groupMaxVal \\
    Y = Y.clamp(min=-limit, max=limit)
    $$

    Where: GroupMax indicates that every 32 elements form a group, and the maximum value within each group is calculated.

  3. Perform quantization

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: symmetric dynamic quantization (pertoken row-wise quantization)
      - Scale factor calculation (row-wise calculation)

        $$
        s_i = \frac{\max_{j \in [0,\ n-1]} |Y_{i,j}|}{C_{\text{MAX}}}
        $$

        Where: $s_i$ is the scale factor of row $i$; $C_{\text{MAX}}$ is the maximum value of the quantization range, which is 127 for int8 and 7 for quint4x2.
      - Quantization computation

        $$
        y_{i,j} = \frac{Y_{i,j}}{s_i}
        $$

    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>: MX quantization

      - Scenario 1, when scaleAlg is 0:
        - Group the input x along the axis dimension into groups of k = 32 numbers. Each group of k numbers  $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized to $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = 32

          $$
          shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
          mxscale = 2^{shared\_exp}\\
          P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space 32\\
          $$

        - The quantized $P_{i}$ values form the output yOut according to the positions of the corresponding $V_{i}$, and mxscale values form the output mxscaleOut according to the corresponding groups along the axis dimension.

        - emax: The exponent of the maximum normal number of the corresponding data type.

            | DataType | Emax |
            | :-----------: | :--: |
            |  FLOAT4_E2M1  |  2   |
            | FLOAT8_E4M3FN |  8   |
            |  FLOAT8_E5M2  |  15  |

      - Scenario 2: When scaleAlg is 1, only FP8 types are involved:
        - Divide the long vector into blocks, each with a length of k. Compute a block scale factor $S_{fp32}^b$ for each block individually, and then map all elements within the block to the target low-precision type FP8 using the same $S_{fp32}^b$. If the last block has fewer than k elements, treat the missing values as 0 and process it as a complete block.
        - Find the maximum absolute value in the block:

          $$
          Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
          $$

        - Map FP32 to the representable range of the target data type FP8, where $Amax(DType)$ is the maximum value that the target precision can represent.

          $$
          S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
          $$

        - Convert the block scale factor $S_{fp32}^b$ to the representable scale value $S_{ue8m0}^b$ in FP8 format.
        - Extract the unbiased exponent $E_{int}^b$ and mantissa $M_{fixp}^b$ from the block floating-point scale factor $S_{fp32}^b$.
        - To ensure no overflow during quantization, round up the exponent and keep it within the representable range of FP8:

          $$
          E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number, and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a subnormal number, and} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{ otherwise} \end{cases}
          $$

        - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
        - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
        - The final step applied for quantization: for each element in a block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$. The final quantization output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ denotes the scale factor of the block, that is, $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ denotes the quantized data in the block.

      - Use case 3: when scaleAlg is 2, only the FP4_E2M1 type is involved:
        - When dstTypeMax = 6.0/7.0:
          - Group the input x along the axis dimension into groups of k = 32 values. Each group of k values  $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized to $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = 32:

            $$
            shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{ if} the one/two bits preceding the high bits of the mantissa \text{ are 1, and the mantissa is not all 0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{ otherwise} \end{cases} \\
            $$

            $$
            P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space 32\\
            $$

          - The quantized $P_{i}$ values form the output yOut according to the positions of the corresponding $V_{i}$, and mxscale values form the output mxscaleOut according to the corresponding groups along the axis dimension.
        - When dstTypeMax != 6.0/7.0:
          - Divide the long vector into blocks, each of length k. Compute a block scaling factor $S_{fp32}^b$ for each block separately, and then map all elements in the block to the target low-precision type using the same $S_{fp32}^b$. If the last block contains fewer than k elements, treat the missing values as 0 and process the block as a complete one.
          - Find the maximum absolute value in the block:

            $$
            Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
            $$

          - Map FP32 to the range representable by the target data type. When dst_max_value=0, $Amax(DType)$ is the maximum value representable by the target precision; when dst_max_value!=0, $Amax(DType)$ is the value passed in by dst_max_value.

            $$
            S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
            $$

          - Convert the block scaling factor $S_{fp32}^b$ to the scale value $S_{ue8m0}^b$ representable in FP4_E2M1 format.
          - Extract the unbiased exponent $E_{int}^b$ and mantissa $M_{fixp}^b$ from the block's floating-point scaling factor $S_{fp32}^b$.
          - To prevent overflow during quantization, round up the exponent, keeping it within the range representable by FP4_E2M1:

            $$
            E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number, and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\ E_{int}^b, & \text{ otherwise} \end{cases}
            $$

          - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
          - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
          - The final step of quantization: for each element in the block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$. The final quantized output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ denotes the block scaling factor, that is, $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ denotes the quantized data in the block.
          - The quantized $P_{i}$ values form the output yOut according to the positions of the corresponding $V_{i}$, and mxscale values form the output mxscaleOut according to the corresponding groups along the axis dimension.

    <!-- end id8 -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call "aclnnRotateQuantGetWorkspaceSize" to obtain the input parameters and calculate the required workspace size based on the process. Then, call "aclnnRotateQuant" to perform the computation.

```c++
aclnnStatus aclnnRotateQuantGetWorkspaceSize(
  const aclTensor   *x,
  const aclTensor   *rotation,
  const aclTensor   *alpha,
  int64_t            axis,
  char              *roundMode,
  int64_t            scaleAlg,
  double             dstTypeMax,
  bool               trans,
  aclTensor         *yOut,
  aclTensor         *scaleOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnRotateQuant(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnRotateQuantGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1491px"><colgroup>
  <col style="width: 162px">
  <col style="width: 121px">
  <col style="width: 332px">
  <col style="width: 200px">
  <col style="width: 275px">
  <col style="width: 118px">
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
      <th>Non-Contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x(aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor to be rotated and quantized.</td>
      <td>Empty tensors are supported.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rotation(aclTensor*)</td>
      <td>Input</td>
      <td>Rotation matrix.</td>
      <td>Supports empty tensors.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2 or 3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha(aclTensor*)</td>
      <td>Input</td>
      <td>Scale factor of the range to be clamped.</td>
      <td>Optional. The shape is (1,) and the data type is BFLOAT16. The valid value range is [0.0, 1.0]. If the value is out of the valid range, clamp is not performed.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis(int64_t)</td>
      <td>Input</td>
      <td>Indicates the axis where quantization is performed (reduce axis).</td>
      <td>Currently, only -1 or the positive index of the last dimension is supported.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundMode(char*)</td>
      <td>Input</td>
      <td>Represents the mode of data conversion.</td>
      <td>Supported values: "rint", "round", and "floor". If a null pointer is passed in, the "rint" mode is used.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg(int64_t)</td>
      <td>Input</td>
      <td>Indicates the scale calculation algorithm.</td>
      <td>Values: 0, 1, 2. Default value: 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax(double)</td>
      <td>Input</td>
      <td>Indicates the maximum value of the quantization target type.</td>
      <td>Supported values are 0.0 or values in the range [6.0, 12.0] (indicating that Amax is the input value).</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>trans(bool)</td>
      <td>Input</td>
      <td>Indicates whether output y is transposed.</td>
      <td>Currently, only false is supported.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut(aclTensor*)</td>
      <td>Output</td>
      <td>Quantized output tensor.</td>
      <td>The output tensor must be pre-allocated. The data type of yOut is the quantization output type (y_dtype). In torch single-operator calls, y_dtype is passed in as a parameter. In aclnn calls, y_dtype does not need to be passed in separately, and the dtype can be obtained from yOut.</td>
      <td>FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2, INT8, INT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOut(aclTensor*)</td>
      <td>Output</td>
      <td>Scale factor computed by dynamic quantization.</td>
      <td>The output tensor must be pre-allocated.</td>
      <td>FLOAT8_E8M0, FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize(uint64_t)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor(aclOpExecutor*)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 288px">
  <col style="width: 125px">
  <col style="width: 742px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input contains a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type, dimension, shape, or parameter value of the input does not meet the constraints.</td>
    </tr>
  </tbody>
  </table>

## aclnnRotateQuant

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 849px">
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
      <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnRotateQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

<!-- npu="950" id9 -->
- <term>Ascend 950PR&950DT products</term>:
  - The shape of x is (*, N), with dimensions in the range [1, 7]; the shape of rotation is (K, K) or (N/K, K, K), with dimensions in the range [2, 3]. In the current version, K only supports 32, 64, and 128.
  - The length (N) of the last dimension of x must be an integer multiple of K.
  - The output type of yOut is FLOAT4_E2M1, FLOAT8_E4M3FN, or FLOAT8_E5M2, and the shape is the same as that of x.
  - x and rotation must have the same data type, and both must be BFLOAT16 or FLOAT16.
  - The lengths of the last two dimensions of rotation must be the same.
  - The shape of scaleOut must be (*, CeilDiv(N,64), 2). The number of dimensions of the shape ranges from 2 to 8, and the data type is FLOAT8_E8M0.
  - The data type of yOut must be in [FLOAT4_E2M1,FLOAT8_E4M3FN,FLOAT8_E5M2], and the shape of yOut must be the same as that of x.
  - alpha is an optional input. If it is not a null pointer, the shape is (1,), the data type is BFLOAT16, and the valid value range is [0.0, 1.0]. If a null pointer is passed in or the value is out of the valid range, no clamp operation is performed.
  - Currently, only -1 or D-1 is supported for axis, where D is the number of dimensions of the shape of x.
  - roundMode supports "rint", "round", and "floor". If a null pointer is passed in, the "rint" mode is used. When the data type of yOut is FLOAT8_E4M3FN or FLOAT8_E5M2, only "rint" is supported for roundMode.
  - The valid values of scaleAlg are 0, 1, and 2. When the data type of yOut is FLOAT8_E4M3FN or FLOAT8_E5M2, only 0 and 1 are supported; when the data type of yOut is FLOAT4_E2M1, only 0 and 2 are supported.
  - dstTypeMax: When scaleAlg=2, dstTypeMax must be in the range [6.0, 12.0]. In other use cases, only 0.0 is supported.
  - trans currently supports only false.

<!-- end id9 -->
<!-- npu="A3,910b" id10 -->
- x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
  - The shape of x is (M, N), and the shape of rotation is (K, K).
  - The shape of rotation must be a square matrix (K, K).
  - The length (N) of the second dimension of x must be a multiple of K, and N must be divisible by 8.
  - When the output data type of yOut is int8, the shape of y must be the same as the shape of x (M, N).
  - When the output data type of yOut is int32, the shape of y must be (M, N//8).
  - x and rotation must have the same data type.
  - The shape of scaleOut must be (M), and the data type must be FLOAT.
  - alpha is an optional input. In the current version, only a null pointer can be passed.
  - axis currently supports only -1 or 1.
  <!-- npu="A3,910b" id11 -->
  - On <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: roundMode currently supports only rint.
  <!-- end id11 -->
  - scaleAlg supports only 0.
  - dstTypeMax supports only 0.0.
  - trans supports only false.
  - The value range of N is [128, 16000].
  - The value range of K is [16, 1024].
  - scaleOut supports only 1D shapes.

<!-- end id10 -->
- Deterministic computation:
  - aclnnRotateQuant defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include <random>

#include "acl/acl.h"
#include "aclnnop/aclnn_rotate_quant.h"

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

int Init(int32_t deviceId, aclrtStream* stream)
{
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
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

std::vector<uint16_t> GenerateRandomBf16Data(int64_t size, unsigned int seed = 42)
{
    std::vector<uint16_t> data(size);
    std::mt19937 gen(seed);
    for (int64_t i = 0; i < size; i++) {
        int sign = (gen() % 2) ? 0x8000 : 0;
        int exp = 0x3F00 + (gen() % 2);
        int mant = gen() % 128;
        data[i] = sign | exp | mant;
    }
    return data;
}

std::vector<uint16_t> GenerateIdentityMatrix(int64_t K)
{
    std::vector<uint16_t> matrix(K * K, 0);
    uint16_t bf16One = 0x3F80;
    for (int64_t i = 0; i < K; i++) {
        matrix[i * K + i] = bf16One;
    }
    return matrix;
}

int main()
{
    // 1. (boilerplate) device/stream initialization. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;

    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Customize the construction based on the API definition.
    // x: (M, N), rot: (K, K), y: (M, N), scale: (M,)
    // Constraints: rot must be square, N % K == 0, N % 8 == 0
    int64_t M = 1024;
    int64_t N = 256;
    int64_t K = 64;

    std::vector<int64_t> xShape = {M, N};
    std::vector<int64_t> rotShape = {K, K};
    std::vector<int64_t> yShape = {M, N};
    std::vector<int64_t> scaleShape = {M};

    // Create x aclTensor (BF16).
    auto xHostData = GenerateRandomBf16Data(M * N, 42);
    void* xDeviceAddr = nullptr;
    aclTensor* xTensor = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &xTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(xTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create x tensor failed.\n"); return ret);

    // Create rot aclTensor (BF16, must be a square matrix).
    auto rotHostData = GenerateIdentityMatrix(K);
    void* rotDeviceAddr = nullptr;
    aclTensor* rotTensor = nullptr;
    ret = CreateAclTensor(rotHostData, rotShape, &rotDeviceAddr, aclDataType::ACL_BF16, &rotTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> rotTensorPtr(rotTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> rotAddrPtr(rotDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create rot tensor failed.\n"); return ret);

    // Create y aclTensor (INT8).
    std::vector<int8_t> yHostData(M * N, 0);
    void* yDeviceAddr = nullptr;
    aclTensor* yTensor = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT8, &yTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(yTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create y tensor failed.\n"); return ret);

    // Create scale aclTensor (FLOAT32).
    std::vector<float> scaleHostData(M, 0.0f);
    void* scaleDeviceAddr = nullptr;
    aclTensor* scaleTensor = nullptr;
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scaleTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(scaleTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scaleAddrPtr(scaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create scale tensor failed.\n"); return ret);

    // 3. Call the CANN operator library API. Change it to the specific API name.
    // Call the first-phase API of aclnnRotateQuant.
    int64_t axis = -1;
    char* roundMode = const_cast<char*>("rint");
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnRotateQuantGetWorkspaceSize(xTensor, rotTensor, nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans,
                                            yTensor, scaleTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
                   LOG_PRINT("aclnnRotateQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // Call the second-phase API of aclnnRotateQuant.
    ret = aclnnRotateQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotateQuant failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values, and copy the results from device memory to host memory. Modify them according to the specific API definition.
    auto ySize = GetShapeSize(yShape);
    std::vector<int8_t> yResult(ySize, 0);
    ret = aclrtMemcpy(yResult.data(), ySize * sizeof(int8_t), yDeviceAddr, ySize * sizeof(int8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y result failed.\n"); return ret);

    auto scaleSize = GetShapeSize(scaleShape);
    std::vector<float> scaleResult(scaleSize, 0.0f);
    ret = aclrtMemcpy(scaleResult.data(), scaleSize * sizeof(float), scaleDeviceAddr, scaleSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale result failed.\n"); return ret);

    // Print part of the results.
    for (int64_t i = 0; i < 5 && i < static_cast<int64_t>(yResult.size()); i++) {
        LOG_PRINT("y[%ld] = %d\n", i, static_cast<int>(yResult[i]));
    }
    for (int64_t i = 0; i < 5 && i < static_cast<int64_t>(scaleResult.size()); i++) {
        LOG_PRINT("scale[%ld] = %f\n", i, scaleResult[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify them according to the specific API definition.
    aclDestroyTensor(xTensor);
    aclDestroyTensor(rotTensor);
    aclDestroyTensor(yTensor);
    aclDestroyTensor(scaleTensor);

    // 7. Release device resources. Modify them according to the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(rotDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
