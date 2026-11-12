# aclnnRmsNormDynamicMxQuant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:30:48.200Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not Supported
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

- Description: The RmsNorm operator is a normalization operation commonly used in large models. Compared with the LayerNorm operator, it removes the mean subtraction part. The DynamicMxQuant operator performs dynamic MX quantization in groups of blocksize along the last axis. The RmsNormDynamicMxQuant operator fuses the DynamicMxQuant operator with the RmsNorm normalization output to reduce move-in and move-out operations. On the last axis of the input, for each group of blocksize=32 numbers, the operator calculates the quantization scale mxscale of the group, divides each number in the group by mxscale, converts the result to the corresponding dst_type based on round_mode, and obtains the quantization result y. When dst_type is FLOAT8_E4M3FN or FLOAT8_E5M2, the value of scale_alg specifies the algorithm for calculating mxscale.
- Formula:

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$
  Use case 1: When scaleAlg is 0:

   - Group the RmsNorm output y along the last axis into groups of k = 32 values. Each group of k values $\{V_i\}_{i=1}^{k}$ is dynamically quantized to $\{mxscale,\{P_i\}_{i=1}^{k}\}$
    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
    $$
    - emax: The exponent of the maximum normal number of the corresponding data type.

        | DataType | Emax |
        | :-----------: | :--: |
        |  FLOAT4_E2M1  |  2   |
        |  FLOAT4_E1M2  |  0   |
        | FLOAT8_E4M3FN |  8   |
        |  FLOAT8_E5M2  |  15  |

  Scenario 2: When scaleAlg is 1, only FP8 types are involved:
    - Divide the long vector into blocks, each of length k, compute a block scaling factor $S_{fp32}^b$ for each block independently, and then map all elements in the block to the target low-precision type FP8 using the same $S_{fp32}^b$.
    - Find the maximum absolute value in the block:

      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$

    - Map FP32 to the range representable by the target data type FP8:

      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$

    - Convert to the scale value representable in FP8 format: $S_{ue8m0}^b$
    - Extract the unbiased exponent $E_{int}^b$ and mantissa $M_{fixp}^b$ from the block floating-point scale factor $S_{fp32}^b$.
    - To avoid overflow during quantization, round up the exponent:

      $
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, and } E_{int}^b < 254 \text{ and } M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a subnormal number, and } M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{otherwise} \end{cases}
      $

    - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
    - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - Apply the final step to quantization: $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call `aclnnRmsNormDynamicMxQuantGetWorkspaceSize` first to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call `aclnnRmsNormDynamicMxQuant` to perform the computation.

```Cpp
aclnnStatus aclnnRmsNormDynamicMxQuantGetWorkspaceSize(
  const aclTensor   *x,
  const aclTensor   *gamma,
  const aclTensor   *beta,
  double             epsilon,
  int64_t            scaleAlg,
  char              *roundMode,
  int64_t            dstType,
  bool               outputRstd,
  aclTensor         *yOut,
  aclTensor         *mxscaleOut,
  aclTensor         *rstdOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnRmsNormDynamicMxQuant(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnRmsNormDynamicMxQuantGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
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
        <th>Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Source data tensor in the normalization process. Corresponds to x in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>For details, see Constraints.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Weight tensor in the normalization process. Corresponds to gamma in the formula.</td>
      <td><ul><li>Support empty Tensor.</li><li>The shape must be one-dimensional and match the last dimension of x.</li><li>The data type defaults to the same as input x; if inconsistent, explicitly set it to FLOAT32.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the bias term in the standardization process. Corresponds to beta in the formula.</td>
      <td><ul><li>Support empty Tensor.</li><li>Optional parameter. Passing in a null pointer is supported.</li><li>If provided, the shape and data type must keep consistent with gamma.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>Value added to the denominator to ensure numerical stability. Corresponds to epsilon in the formula.</td>
      <td><ul><li>It is recommended to set a small positive value, such as 1e-6.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg (int64_t)</td>
      <td>Input</td>
      <td>Represents the calculation method of mxscaleOut, which is scaleAlg in the corresponding formula.</td>
      <td><ul><li>Value Range is 0 or 1. 0 indicates use case 1, and 1 indicates use case 2.</li><li>If dstType is FLOAT4_E2M1/FLOAT4_E1M2, only 0 is supported.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundMode (char*)</td>
      <td>Input</td>
      <td>Specifies the mode of data conversion, corresponding to round_mode in the formula.</td>
      <td><ul><li>Empty pointers are not supported. A supported string must be explicitly passed in.</li><li>When dstType is 40/41, {"rint", "floor", "round"} are supported.</li><li>When dstType is 36/35, only {"rint"} is supported.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Specifies the type of yOut after data conversion, corresponding to DType in the formula.</td>
      <td><ul><li>The value range is {35, 36, 40, 41}, corresponding to {35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2}.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputRstd (bool)</td>
      <td>Input</td>
      <td>Specifies whether to output a valid rstdOut.</td>
      <td><ul><li>Supports True and False.</li><li>If outputRstd is False, rstdOut is an invalid output.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the normalized and quantized result, corresponding to Pi and di in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape must be the same as that of input x.</li><li>The data type must be consistent with dstType.</li></ul></td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the quantization scale corresponding to each group, corresponding to mxscale and Sb in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>On the trailing axis, the shape is the corresponding value of x divided by blocksize=32, rounded up, and padded to an even number. The fill value is 0.</li><li>The remaining dimensions Keep Consistent with the first few dimensions of input x. The first few dimensions refer to the dimensions of x minus the dimensions of gamma, indicating the dimensions that do not need norm.</li><li>See Constraints for the detailed calculation process.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the reciprocal of the standard deviation after normalization. The reciprocal of Rms(x) in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>When outputRstd is True, the number of dimensions is kept consistent with input `x`. All dimensions except the last one are kept consistent with the corresponding dimensions of `x`, and the last dimension is 1. Example of the relationship between the `rstdOut` shape and the `x` shape: if the `x` shape is (2,3,4,8), the `rstdOut` shape is (2,3,4,1).</li><li>When outputRstd is False, the final output of this parameter is invalid.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
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

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>The input x, gamma, yOut, and mxscaleOut are null pointers.</td>
    </tr>
    <tr>
      <td>When outputRstd is True, the passed-in rstdOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_INNER_NULLPTR</td>
      <td rowspan="8">561103</td>
      <td>The dimension count and data type of the input or output parameters are out of range, or the data type of dstType does not match that of yOut.</td>
    </tr>
    <tr>
      <td>The shapes of input x and output yOut are not exactly the same.</td>
    </tr>
    <tr>
      <td>The shapes of gamma and beta (if present) are not exactly the same, or the dimension count of either is not 1, or the axis length is not equal to the size of the last axis of x, or their types are different.</td>
    </tr>
    <tr>
      <td>The dimensions of gamma do not match the dimensions of x to be normalized, or the dimensions of rstdOut do not match the dimensions of x not to be normalized, or the axis length of rstdOut corresponding to the normalized dimensions of x is not 1.</td>
    </tr>
    <tr>
      <td>scaleAlg is not 0 or 1, or roundMode is a null pointer, or roundMode is not one of {"rint", "floor", "round"}.</td>
    </tr>
    <tr>
      <td>When dstType is FP8, roundMode is not "rint".</td>
    </tr>
    <tr>
      <td>When dstType is fp4, scaleAlg is not 0, or the size of the last axis of input x is not divisible by 2.</td>
    </tr>
    <tr>
      <td>The number of dimensions of mxscaleOut is not equal to the number of dimensions of input x + 1, or the axis length does not comply with Constraints.</td>
    </tr>
  </tbody></table>

## aclnnRmsNormDynamicMxQuant

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1042px"><colgroup>
  <col style="width: 141px">
  <col style="width: 110px">
  <col style="width: 791px">
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
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device, obtained by calling the first-phase API aclnnRmsNormDynamicMxQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream where the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Boundary Value Scenario Description**
  - When the input is Inf: 1) output yOut is 0; 2) output mxscaleOut is 255, and the fill value of even pads is 0; 3) output rstdOut is 0.
  - When the input is NaN: 1) output yOut is 0; 2) output mxscaleOut is 255, and the fill value of even pads is 0; 3) output rstdOut is NaN.

- **Data Format Description**

  It is recommended to use the ND format for all input and output tensors. Other data formats are converted to the ND format by the framework by default for processing.

- **Supported Data Types by Product Model**

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:

    | `x` Data Type | `gamma` Data Type | `beta` Data Type | `yOut` Data Type | `mxscaleOut` Data Type | `rstdOut` Data Type |
    | ----------- | -------------- | ------------- | -------------- | ------------------- | ---------------- |
    | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT4_E2M1    | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT4_E1M2    | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT8_E4M3FN  | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT8_E5M2    | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16       | BFLOAT16      | FLOAT4_E2M1    | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16       | BFLOAT16      | FLOAT4_E1M2    | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16       | BFLOAT16      | FLOAT8_E4M3FN  | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16       | BFLOAT16      | FLOAT8_E5M2    | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT4_E2M1    | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT4_E1M2    | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT8_E4M3FN  | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT8_E5M2    | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | FLOAT32        | FLOAT32       | FLOAT4_E2M1    | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | FLOAT32        | FLOAT32       | FLOAT4_E1M2    | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | FLOAT32        | FLOAT32       | FLOAT8_E4M3FN  | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | FLOAT32        | FLOAT32       | FLOAT8_E5M2    | FLOAT8_E8M0         | FLOAT32          |
  <!-- end id7 -->

- The shape constraints of mxscaleOut are as follows:
  - rank(mxscaleOut) = rank(x) + 1.
  - mxscaleOut.shape[-2] = (ceil(x.shape[-1] / 32) + 2 - 1) / 2.
  - mxscaleOut.shape[-1] = 2.
  - Other dimensions are consistent with input x.

- The shape constraints of x are as follows:
  - If the data type of output yOut is FLOAT4_E2M1 or FLOAT4_E1M2, the value of the last dimension of x must be an even number.

- Deterministic computation:
  - aclnnRmsNormDynamicMxQuant defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm_dynamic_mx_quant.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
      aclrtResetDevice(deviceId); aclFinalize(); return ret);
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the acl API reference.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. The construction must be customized based on the API.
  std::vector<int64_t> xShape = {2, 64};
  std::vector<int64_t> gammaShape = {64};
  std::vector<int64_t> mxscaleShape = {2, 1, 2};  // CeilDiv(CeilDiv(64, 32), 2) = 1
  std::vector<int64_t> rstdShape = {2, 1};

  void* xDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;

  void* yDeviceAddr = nullptr;
  void* mxscaleDeviceAddr = nullptr;
  void* rstdDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* y = nullptr;
  aclTensor* mxscale = nullptr;
  aclTensor* rstd = nullptr;

  int64_t xShapeSize = GetShapeSize(xShape);
  int64_t gammaShapeSize = GetShapeSize(gammaShape);
  int64_t mxscaleShapeSize = GetShapeSize(mxscaleShape);
  int64_t rstdShapeSize = GetShapeSize(rstdShape);

  std::vector<short> xHostData(xShapeSize, 0x3800);
  std::vector<short> gammaHostData(gammaShapeSize, 0x3e00);
  std::vector<short> betaHostData(gammaShapeSize, 0);

  std::vector<int8_t> yHostData(xShapeSize, 0);
  std::vector<int8_t> mxscaleHostData(mxscaleShapeSize, 0);
  std::vector<float> rstdHostData(rstdShapeSize, 0);

  float epsilon = 1e-6;
  bool outputRstd = true;  // Set to true to output a valid rstdOut.

  // Create the x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the gamma aclTensor.
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the y aclTensor (FLOAT8_E4M3FN).
  ret = CreateAclTensor(yHostData, xShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the mxscale aclTensor (FLOAT8_E8M0).
  ret = CreateAclTensor(mxscaleHostData, mxscaleShape, &mxscaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the rstd aclTensor.
  ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRmsNormDynamicMxQuant.
  ret = aclnnRmsNormDynamicMxQuantGetWorkspaceSize(
      x, gamma, nullptr, epsilon, 0, nullptr, 36, outputRstd, y, mxscale, rstd, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("aclnnRmsNormDynamicMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory according to the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRmsNormDynamicMxQuant.
  ret = aclnnRmsNormDynamicMxQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("aclnnRmsNormDynamicMxQuant failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result in device memory to the host.
  auto size = GetShapeSize(xShape);
  std::vector<int8_t> yRet(size, 0);
  ret = aclrtMemcpy(
      yRet.data(), yRet.size() * sizeof(yRet[0]), yDeviceAddr, size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, yRet[i]);
  }

  // 6. Release aclTensor.
  aclDestroyTensor(x);
  aclDestroyTensor(gamma);
  aclDestroyTensor(y);
  aclDestroyTensor(mxscale);
  aclDestroyTensor(rstd);

  // 7. Release device resources.
  aclrtFree(xDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(mxscaleDeviceAddr);
  aclrtFree(rstdDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
