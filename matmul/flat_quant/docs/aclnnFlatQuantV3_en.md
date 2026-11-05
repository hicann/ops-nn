# aclnnFlatQuantV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:00:34.210Z -->

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

- **Differences from the V2 version**:
  - The V3 version adds the `groupList` and `groupListType` parameters to specify the quantization grouping.
  - The V2 version adds the `dstTypeMax` parameter to control the maximum value of the target data type in pergroup quantization mode (corresponding to the float4_e2m1 output type).
  - The V1 version does not support customizing the maximum value of the target data type and defaults to the maximum value representable by the target precision.

- **User selection suggestions**:
  - Select the V3 version if you need to specify quantization grouping.
  - Select the V2 and V3 versions if you need to customize the maximum value of the target data type for float4_e2m1 quantization (range 6.0-12.0).
  - If quantization is not grouped, the default maximum value of the target precision and the INT4/INT32 quantized output type are used, all V1, V2, and V3 versions are applicable.
  - For use cases that require finer control over the quantization range to optimize model performance or accuracy, the V2 and V3 versions are recommended.

- Description: This fused operator performs two small matrix multiplications on the input matrix x at a time, that is, right-multiplying by the input matrix kroneckerP2 and left-multiplying by the input matrix kroneckerP1, and then quantizes the matrix multiplication result. Currently, pertoken and pergroup quantization modes are supported, corresponding to the INT4 and FLOAT4_E2M1 quantized output types respectively.

- Matrix multiplication formula:

  1. Input x is right-multiplied by kroneckerP2:

     $$
     x' = x @ kroneckerP2
     $$

  2. Left-multiply x' by kroneckerP1:

     $$
     x'' = kroneckerP1@x'
     $$

- Quantization calculation mode:

  pertoken quantization mode:

  1. Compute the maximum absolute value along dimension 0 of x'' and divide it by (7 / clipRatio) to calculate the quantization scale for quantization to the INT4 format:

     $$
     quantScale = [max(abs(x''[0,:,:])),max(abs(x''[1,:,:])),...,max(abs(x''[M-1,:,:]))]/(7 / clipRatio)
     $$

  2. Calculate the output out:

     $$
     out = x'' / quantScale
     $$

  pergroup quantization mode:
  - When dstTypeMax = 0.0:
    1. After matrix multiplication, x'' has a shape of [M,N1,N2]. To calculate mx_quantize in the pergroup quantization mode, reshape it to [M,N1*N2], denoted as x2.

    2. Group x2 along the second dimension by groupsize. Each group contains elements e0,e1...e31. Calculate emax.

       $$
       emax = max(e0,e1....e31)
       $$

    3. Calculate reduceMaxValue and sharedExp.

       $$
       reduceMaxValue = log2(reduceMax(x2),groupSize=32)
       $$

       $$
       sharedExp[M,N1*N2/32] = reduceMaxValue -emax
       $$

    4. Calculate quantScale.

       $$
       quantScale[M,N1*N2/32] = 2 ^ {sharedExp[M,N1*N2/32]}
       $$

    5. One quantScale is shared per groupsize. Calculate out.

       $$
       out = x2 / quantScale
       $$

  - When dstTypeMax = 6.0/7.0:
    - Group the input x along the axis dimension in groups of k = blocksize numbers, with k numbers per group  $\{\{V_i\}_{i=1}^{k}\}$ dynamically quantized to $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = blocksize:

      $$
      shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{ if} the preceding one/two high bits of the mantissa \text{ are 1, and the mantissa is not all 0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{ otherwise} \end{cases} \\
      $$

      $$
      P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
      $$

    - ​The quantized $P_{i}$ form the output yOut according to the corresponding positions of $V_{i}$, and mxscale forms the output mxscaleOut according to the corresponding grouping on the axis dimension.
  - When dstTypeMax != 0.0/6.0/7.0:
    - Divide the long vector into blocks, each with a length of k. Compute a block scale factor $S_{fp32}^b$ for each block individually, and then map all elements within the block to the target low-precision type FP8 using the same $S_{fp32}^b$. If the last block has fewer than k elements, treat the missing values as 0 and process it as a complete block.
    - Find the maximum absolute value of the values in the block:

      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$

    - Map FP32 to the range representable by the target data type FP8. When dst_max_value=0, $Amax(DType)$ is the maximum value representable by the target precision; when dst_max_value!=0, $Amax(DType)$ is the value passed in for dst_max_value.

      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$

    - Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ representable in FP8 format.
    - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point block scaling factor $S_{fp32}^b$.
    - To ensure no overflow during quantization, round up the exponent and keep it within the representable range of FP8:

      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number, and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\ E_{int}^b, & \text{ otherwise} \end{cases}
      $$

    - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
    - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - Apply to the final quantization step. For each element within a block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$. The final quantized output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ represents the block scale factor, i.e., $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ represents the quantized data within the block.
    - The quantized $P_{i}$ are placed at the positions of the corresponding $V_{i}$ to form the output yOut, and mxscale forms the output mxscaleOut according to the grouping on the corresponding axis dimension.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnFlatQuantV3GetWorkspaceSize` API to obtain the workspace size required for computation and the executor that includes the calculation process of the operator, and then call the `aclnnFlatQuantV3` API to execute the computation.

```Cpp
aclnnStatus aclnnFlatQuantV3GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *kroneckerP1,
  const aclTensor *kroneckerP2,
  const aclTensor *groupListOptional,
  double           clipRatio,
  double           dstTypeMax,
  int64_t          groupListType,
  aclTensor       *out,
  aclTensor       *quantScale,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFlatQuantV3(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnFlatQuantV3GetWorkspaceSize

- **Parameters**:
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
      <td>Raw input data, corresponding to `x` in the formula.</td>
      <td><ul><li>Does not support empty Tensor.</li><li>The shape is [M, N1, N2], where M is not greater than 262144, and both N1 and N2 are not greater than 256.</li><li>If the data type of out is INT32, N2 must be a multiple of 8.</li><li>If the data type of out is INT4, N2 must be an even number.</li><li>If the data type of out is FLOAT4_E2M1, N2 must be an even number.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kroneckerP1 (aclTensor*)</td>
      <td>Input</td>
      <td>Input calculation matrix 1, corresponding to `kroneckerP1` in the formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The shape is [N1, N1], and N1 is the same as the N1 dimension in x.</li><li>The data type is the same as the data type of the input parameter x.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kroneckerP2(aclTensor*)</td>
      <td>Input</td>
      <td>Calculation matrix 2 input, corresponding to `kroneckerP2` in the formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The shape is [N2, N2], and N2 is the same as the N2 dimension in x. The shape [0,0] is supported only in the perGroup use case, indicating that the operator does not use the right matrix for calculation and only performs multiplication with the left matrix.</li><li>The data type is the same as the data type of the input parameter x.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupListOptional (aclTensor*)</td>
      <td>Optional input</td>
      <td>Represents the quantization group size distribution of input `x`.</td>
      <td><ul><li>Does not support empty Tensor.</li><li>When groupListType is 0 or 1, the shape is [G]. When groupListType is 2, the shape is [G, 2]. G indicates the number of groups and must be less than or equal to 1024.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>clipRatio (double)</td>
      <td>Input</td>
      <td>Controls the clipping ratio in quantization, which is `clipRatio` in the corresponding formula.</td>
      <td><ul><li>Input data range: (0, 1].</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax (double)</td>
      <td>Input</td>
      <td>Indicates the value of maxType, corresponding to Amax(DType) in the formula.</td>
      <td><ul><li>Supported values: 0.0 and 6.0-12.0. 0.0 indicates that this parameter is not used; 6.0-12.0 indicates the maximum value of the target data type. This value can be set only when the data type is FLOAT4_E2M1.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType (int64)</td>
      <td>Input</td>
      <td>Indicates the grouping mode of the groupList input.</td>
      <td><ul><li>Supports values 0, 1, and 2 when groupListOptional is not nullptr.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor, corresponding to out in the corresponding formula.</td>
      <td><ul><li>Does not support empty Tensor.</li><li>If the data type is INT32, the last dimension of shape is 1/8 of the last dimension of input parameter x, and the other dimensions are the same as those of x.</li><li>If the data type is INT4, the shape is consistent with the input parameter x.</li><li>If the data type is FLOAT4_E2M1, the shape is [M,N1*N2].</li></ul></td>
      <td>INT4, INT32, FLOAT4_E2M1</td>
      <td>ND</td>
      <td>3 or 2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantScale (aclTensor*)</td>
      <td>Output</td>
      <td>Quantization scale of the output, which is quantScale in the corresponding formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>If the quantized output type is INT4 or INT32, the shape is [M], where M is the same as the M dimension in x, and the data type is FLOAT32.</li><li>If the quantized output type is FLOAT4_E2M1, the shape is [M,ceildiv(N1*N2,64),2], and the data type is FLOAT8_E8M0.</li></ul></td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1 or 3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace that needs to be side-applied by the user on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns an op executor, which includes the calculation process of the operator.</td>
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
      <th>Error code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The required inputs (x, kroneckerP1, kroneckerP2) or required outputs (out, quantScale) in the input parameters are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>The data type or data format of x, kroneckerP1, kroneckerP2, out, or quantScale is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of kroneckerP1 and kroneckerP2 are inconsistent with x.</td>
    </tr>
     <tr>
      <td>The dimension of x is not 3.</td>
    </tr>
    <tr>
      <td>The first dimension of x exceeds the range [1, 262144], or the second dimension exceeds [1, 256], or the third dimension exceeds [1, 256].</td>
    </tr>
    <tr>
      <td>The dimension of kroneckerP1 is not 2, or the first dimension and second dimension do not match the second dimension of x.</td>
    </tr>
    <tr>
      <td>In the perToken use case, the dimension of kroneckerP2 is not 2, or the first dimension and second dimension do not match the third dimension of x.</td>
    </tr>
    <tr>
      <td>In the INT4 or INT32 use case, the dimension of quantScale is not 1, or the first dimension does not match the first dimension of x.</td>
    </tr>
    <tr>
      <td>In the FLOAT4_E2M1 use case, the dimension of quantScale is not 3, or the first dimension does not match the first dimension of x.</td>
    </tr>
    <tr>
      <td>The value of clipRatio is out of the range (0, 1].</td>
    </tr>
    <tr>
      <td>If the data type of out is INT4, the tail axis size of the shape of x is not an even number, or the shape of x does not match the shape of out.</td>
    </tr>
    <tr>
      <td>If the data type of out is INT32, the tail axis of the shape of x is not 8 times the tail axis size of the shape of out, or the sizes of the non-tail axes of the shapes of x and out do not match.</td>
    </tr>
    <tr>
      <td>The data type, dimension, shape, or parameter value of groupList or groupListType does not comply with the constraints.</td>
    </tr>
  </tbody></table>

## aclnnFlatQuantV3

- **Parameters**:

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Address of the workspace memory side-applied on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory side-applied on the Device, obtained by the first-phase API aclnnFlatQuantV3GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, includes the calculation process of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- The parameters groupListOptional and groupListType must meet the following constraints:
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:
    - groupListOptional supports only nullptr.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
    - When groupListOptional is not nullptr:
      - The value of groupListType must be selected from [0, 1, 2].
      - When groupListType is 0 or 1, the shape is [G]. When groupListType is 2, the shape is [G, 2]. G indicates the number of groups and must be less than or equal to 1024.
      - The values of groupListOptional must meet the following conditions. Otherwise, the output cannot be guaranteed to meet expectations:
        - When groupListType is 0, groupListOptional must be a non-negative monotonically non-decreasing sequence, indicating the cumsum result (cumulative sum) of the size of each group after grouping. The last value must be less than or equal to the first dimension of the tensor in x.
        - When groupListType is 1, groupListOptional must be a non-negative sequence, indicating the size of each group after grouping. The sum of the values must be less than or equal to the first dimension of the tensor in x.
        - When groupListType is 2, groupListOptional must be a non-negative sequence. The data layout is [[groupIdx0, groupSize0], [groupIdx1, groupSize1]...], where groupSize indicates the size of each group after grouping. The sum of the values in the second column must be less than or equal to the first dimension of the tensor in x.
  <!-- end id8 -->
- Deterministic computation:
  - aclnnFlatQuantV3 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_flat_quant_v3.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate: resource initialization
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
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to device-side memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual
    // Fill in deviceId according to the actual device
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as required
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output as required by the API
    std::vector<int64_t> xShape = {16, 16, 16};
    std::vector<int64_t> kroneckerP1Shape = {16, 16};
    std::vector<int64_t> kroneckerP2Shape = {16, 16};
    std::vector<int64_t> outShape = {16, 16, 2};
    std::vector<int64_t> quantScaleShape = {16};
    void* xDeviceAddr = nullptr;
    void* kroneckerP1DeviceAddr = nullptr;
    void* kroneckerP2DeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* quantScaleDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* kroneckerP1 = nullptr;
    aclTensor* kroneckerP2 = nullptr;
    aclTensor* out = nullptr;
    aclTensor* quantScale = nullptr;
    double clipRatio = 1.0;
    double dstTypeMax = 0.0;
    int64_t groupListType = 0;
    std::vector<aclFloat16> xHostData(16 * 16 * 16, aclFloatToFloat16(1));
    std::vector<aclFloat16> kroneckerP1HostData(16 * 16, aclFloatToFloat16(1));
    std::vector<aclFloat16> kroneckerP2HostData(16 * 16, aclFloatToFloat16(1));
    std::vector<int32_t> outHostData(16 * 16 * 2, 1);
    std::vector<float> quantScaleHostData(16, 0);
    // Create the x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create kroneckerP1 aclTensor.
    ret = CreateAclTensor(kroneckerP1HostData, kroneckerP1Shape, &kroneckerP1DeviceAddr, aclDataType::ACL_FLOAT16,
                          &kroneckerP1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create kroneckerP2 aclTensor.
    ret = CreateAclTensor(kroneckerP2HostData, kroneckerP2Shape, &kroneckerP2DeviceAddr, aclDataType::ACL_FLOAT16,
                          &kroneckerP2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create quantScale aclTensor.
    ret = CreateAclTensor(quantScaleHostData, quantScaleShape, &quantScaleDeviceAddr, aclDataType::ACL_FLOAT,
                          &quantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Modify it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnFlatQuantV3.
    ret = aclnnFlatQuantV3GetWorkspaceSize(x, kroneckerP1, kroneckerP2, nullptr, clipRatio, dstTypeMax, groupListType,
                                           out, quantScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlatQuantV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnFlatQuantV3.
    ret = aclnnFlatQuantV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlatQuantV3 failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for the task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values and copy the result from device memory to the host side. Modify it according to the specific API definition.
    auto size = GetShapeSize(outShape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    auto quantScaleSize = GetShapeSize(quantScaleShape);
    std::vector<float> quantScaleResultData(quantScaleSize, 0);
    ret = aclrtMemcpy(quantScaleResultData.data(), quantScaleResultData.size() * sizeof(quantScaleResultData[0]),
                      quantScaleDeviceAddr, quantScaleSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < quantScaleSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, quantScaleResultData[i]);
    }

    // 6. Release the aclTensor. Modify it based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(kroneckerP1);
    aclDestroyTensor(kroneckerP2);
    aclDestroyTensor(out);
    aclDestroyTensor(quantScale);

    // 7. Release the device resources. Modify them based on the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(kroneckerP1DeviceAddr);
    aclrtFree(kroneckerP2DeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(quantScaleDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
