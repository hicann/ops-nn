# aclnnLayerNorm&aclnnLayerNormWithImplMode

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:23:54.567Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Performs normalization calculation with a mean of 0 and a standard deviation of 1 on the specified layer. Compared with aclnnLayerNorm, aclnnLayerNormWithImplMode allows you to select different normalization implementation strategies by configuring the implMode parameter, adapting to different use cases and performance requirements.

- Formula:

  $$
  out = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + eps}} * weightOptional + biasOptional
  $$

  $$
  meanOutOptional = \mathrm{E}[x]
  $$

  $$
  rstdOutOptional = \frac{1}{ \sqrt{\mathrm{Var}[x] + eps}}
  $$

  Where E[x] represents the mean of the input, and Var[x] represents the variance of the input.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnLayerNormGetWorkspaceSize" or "aclnnLayerNormWithImplModeGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor containing the operator computation process. Then call "aclnnLayerNorm" or "aclnnLayerNormWithImplMode" to perform the computation.

```Cpp
aclnnStatus aclnnLayerNormGetWorkspaceSize(
  const aclTensor   *input,
  const aclIntArray *normalizedShape,
  const aclTensor   *weightOptional,
  const aclTensor   *biasOptional,
  double             eps,
  aclTensor         *out,
  aclTensor         *meanOutOptional,
  aclTensor         *rstdOutOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnLayerNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

```Cpp
aclnnStatus aclnnLayerNormWithImplModeGetWorkspaceSize(
  const aclTensor    *input,
  const aclIntArray  *normalizedShape,
  const aclTensor    *weightOptional,
  const aclTensor    *biasOptional,
  double              eps,
  aclTensor          *out,
  aclTensor          *meanOutOptional,
  aclTensor          *rstdOutOptional,
  int32_t             implMode,
  uint64_t           *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnLayerNormWithImplMode(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnLayerNormGetWorkspaceSize

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
      <td>input (aclTensor*)</td>
      <td>Input</td>
      <td>Input for the normalization calculation, corresponding to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape is [A1,...,Ai,R1,...,Rj], where A1 to Ai are the dimensions that do not require norm, and R1 to Rj are the dimensions that require norm.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>normalizedShape (aclIntArray*)</td>
      <td>Input</td>
      <td>Dimensions that require the norm calculation.</td>
      <td>The value is [R1,...,Rj], whose length is no more than the shape length of the input input. It cannot be empty.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input, which is the weight for the normalization calculation. It corresponds to `weightOptional` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>If `weightOptional` is not empty:<ul><li>The data type is the same as that of the input `input` or is FLOAT32. If `biasOptional` exists, `weightOptional` and `biasOptional` have the same data type.</li><li>The shape is the same as that of `normalizedShape`, which is [R1,...,Rj].</li></ul></li><li>If `weightOptional` is empty, a tensor with the shape [R1,...,Rj] and all values 1 is constructed inside the interface.<ul><li>If `biasOptional` exists, `weightOptional` and `biasOptional` have the same data type.</li><li>If `biasOptional` does not exist, `weightOptional` has the same data type as the input `input`.</li></ul></li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input parameter that indicates the offset for the normalization calculation. Corresponds to `biasOptional` in the corresponding formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>When `biasOptional` is not empty:<ul><li>The data type must be the same as that of input `input` or be FLOAT. If `weightOptional` exists, `biasOptional` and `weightOptional` must have the same data type.</li><li>The shape must be the same as `normalizedShape`, that is, [R1,...,Rj].</li></ul></li><li>When `biasOptional` is empty, a tensor with the shape [R1,...,Rj] and all-zero data is constructed inside the interface.<ul><li>If `weightOptional` exists, `biasOptional` and `weightOptional` have the same data type.</li><li>If `weightOptional` does not exist, `biasOptional` has the same data type as input `input`.</li></ul></li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>eps (double)</td>
      <td>Input</td>
      <td>Value added to the denominator to ensure numerical stability. Corresponds to `eps` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the result of the normalization calculation. Corresponds to `out` in the corresponding formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape must be the same as the shape of `input`, that is, [A1,...,Ai,R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>meanOutOptional (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output. Represents the mean after normalization. Corresponds to `meanOutOptional` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>If `rstdOutOptional` exists, its shape is the same as that of `rstdOutOptional`. The shape is [A1,...,Ai,1,...,1], where there are j 1s after Ai, and j is the same as the axis length to be normalized.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOutOptional (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output, indicating the reciprocal of the standard deviation after normalization. Corresponds to `rstdOutOptional` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>If `meanOutOptional` exists, its shape is the same as that of `meanOutOptional`. The shape is [A1,...,Ai,1,...,1], where there are j 1s after Ai, and j is the same as the axis length to be normalized.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910,310p,310b" id7 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>, <term>Atlas 200I/500 A2 inference products</term>: The data type of the parameters input, weightOptional, biasOptional, out, meanOutOptional, and rstdOutOptional does not support BFLOAT16.
  <!-- end id7 -->

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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input, normalizedShape, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>For input, normalizedShape, weightOptional (when not null), biasOptional (when not null), out, meanOutOptional (when not null), and rstdOutOptional (when not null), the shape dimension exceeds 8 dimensions.</td>
    </tr>
    <tr>
      <td>For input, weightOptional (when not null), biasOptional (when not null), out, meanOutOptional (when not null), and rstdOutOptional (when not null), the data type is not in the supported range.</td>
    </tr>
    <tr>
      <td>The number of dimensions of normalizedShape is less than 1.</td>
    </tr>
    <tr>
      <td>weightOptional is not null and its shape is different from normalizedShape.</td>
    </tr>
    <tr>
      <td>biasOptional is not null and its shape is different from normalizedShape.</td>
    </tr>
    <tr>
      <td>The number of dimensions of input is less than that of normalizedShape.</td>
    </tr>
    <tr>
      <td>When the shape of input is right-aligned with normalizedShape, the shapes of the corresponding dimensions are not equal.</td>
    </tr>
    <tr>
      <td>The shapes of input and out are inconsistent.</td>
    </tr>
  </tbody></table>

## aclnnLayerNorm

- **Parameters**

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device. Obtained by the first-phase API aclnnLayerNormGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnLayerNormWithImplModeGetWorkspaceSize

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
      <td>input (aclTensor*)</td>
      <td>Input</td>
      <td>Represents the input for the normalization calculation, corresponding to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape is [A1,...,Ai,R1,...,Rj], where A1 to Ai represent dimensions that do not require norm, and R1 to Rj represent dimensions that require norm.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>normalizedShape (aclIntArray*)</td>
      <td>Input</td>
      <td>Represents the dimensions that require norm calculation.</td>
      <td>The value is [R1,...,Rj]. Its length must be less than or equal to the shape length of input, and it cannot be empty.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input parameter, indicating the weight used in the normalization calculation. Corresponds to `weightOptional` in the corresponding formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>When `weightOptional` is not empty:<ul><li>The data type is the same as that of input `input`, or is FLOAT32. When `biasOptional` exists, `weightOptional` and `biasOptional` have the same data type.</li><li>The shape is [R1,...,Rj], the same as `normalizedShape`.</li></ul></li><li>When `weightOptional` is empty, a tensor with shape [R1,...,Rj] and all elements set to 1 is constructed inside the interface.<ul><li>When `biasOptional` exists, `weightOptional` and `biasOptional` have the same data type.</li><li>When `biasOptional` does not exist, `weightOptional` has the same data type as input `input`.</li></ul></li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional(aclTensor*)</td>
      <td>Input</td>
      <td>Optional input. Indicates the offset for the normalization calculation. It corresponds to `biasOptional` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>When `biasOptional` is not empty:<ul><li>The data type must be the same as that of the input `input` or be FLOAT32. If `weightOptional` exists, `biasOptional` and `weightOptional` must have the same data type.</li><li>The shape must be [R1,...,Rj], the same as that of `normalizedShape`.</li></ul></li><li>When `biasOptional` is empty, a tensor with a shape of [R1,...,Rj] and all elements set to 0 is constructed inside the interface.<ul><li>If `weightOptional` exists, `biasOptional` and `weightOptional` must have the same data type.</li><li>If `weightOptional` does not exist, `biasOptional` must have the same data type as the input `input`.</li></ul></li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>eps (double)</td>
      <td>Input</td>
      <td>Value added to the denominator to ensure numerical stability. Corresponds to `eps` in the corresponding formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the result of the normalization calculation. It is `out` in the corresponding formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape must be the same as the shape of `input`, which is [A1,...,Ai,R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>meanOutOptional (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output. Indicates the mean after normalization. It is `meanOutOptional` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is [A1,...,Ai,1,...,1], where there are j 1s after Ai, consistent with the axis length to be normalized.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOutOptional (aclTensor*)</td>
      <td>Output</td>
      <td>Optional output, indicating the reciprocal of the standard deviation after normalization. This parameter corresponds to `rstdOutOptional` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape is [A1,...,Ai,1,...,1], where there are j 1s after Ai, consistent with the axis length to be normalized.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>implMode (int32_t)</td>
      <td>Input</td>
      <td>Specifies the computation mode for kernel selection.</td>
      <td>Supports 0, 1, and 2. The default value is 0. 0 indicates high-precision mode, 1 indicates high-performance mode, and 2 indicates the FLOAT16 computation mode. Use the high-performance mode with caution because it causes precision loss. The FLOAT16 computation mode is supported only in use cases where all inputs are FLOAT16 and provides the lowest computation precision.</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator calculation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910,310p,310b" id8 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>, <term>Atlas 200I/500 A2 inference products</term>: The data type of the parameters input, weightOptional, biasOptional, out, meanOutOptional, and rstdOutOptional does not support BFLOAT16.
  <!-- end id8 -->

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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>input, normalizedShape, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>The shape of input, normalizedShape, weightOptional (if not null), biasOptional (if not null), out, meanOutOptional (if not null), or rstdOutOptional (if not null) exceeds 8 dimensions.</td>
    </tr>
    <tr>
      <td>The data type of input, weightOptional (if not null), biasOptional (if not null), out, meanOutOptional (if not null), or rstdOutOptional (if not null) is not in the supported range.</td>
    </tr>
    <tr>
      <td>The dimension of normalizedShape is less than 1.</td>
    </tr>
    <tr>
      <td>weightOptional is not null and its shape is different from normalizedShape.</td>
    </tr>
    <tr>
      <td>biasOptional is not empty and its shape is not equal to that of normalizedShape.</td>
    </tr>
    <tr>
      <td>input has fewer dimensions than normalizedShape.</td>
    </tr>
    <tr>
      <td>The shapes of corresponding dimensions are not equal when the shape of input is right-aligned with normalizedShape.</td>
    </tr>
    <tr>
      <td>The value of implMode is not within the range of 0, 1, or 2.</td>
    </tr>
    <tr>
      <td>The value of implMode is 2 and the input data types are not all FLOAT16.</td>
    </tr>
    <tr>
      <td>The shapes of input and out are inconsistent.</td>
    </tr>
  </tbody></table>

## aclnnLayerNormWithImplMode

- **Parameters**

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
      <td>Workspace memory address applied for on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied for on the Device, obtained by the first-phase API aclnnLayerNormWithImplModeGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
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

- The shape of input, normalizedShape, weightOptional (when non-null), biasOptional (when non-null), out, meanOutOptional (when non-null), or rstdOutOptional (when non-null) does not exceed 8 dimensions.
- Deterministic computation:
  - aclnnLayerNorm defaults to a deterministic implementation.
  - aclnnLayerNormWithImplMode defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

- aclnnLayerNorm example:

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_layer_norm.h"

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
      // Boilerplate for resource initialization
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
  int CreateAclTensorMem(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  void aclCreateTensorP(const std::vector<T>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
  {
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
  }

  template <typename T>
  void aclCreateIntArrayP(const std::vector<T>& hostData, aclIntArray** intArray)
  {
      *intArray = aclCreateIntArray(hostData.data(), hostData.size());
  }

  int main()
  {
      // 1. (Boilerplate) Initialize device/stream. For details, refer to the acl API manual.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output, customized based on the API definition.
      std::vector<int64_t> xShape = {1, 2, 32};
      std::vector<int64_t> normShape = {32};
      std::vector<int64_t> meanShape = {1, 2, 1};
      void* xDeviceAddr = nullptr;
      void* weightDeviceAddr = nullptr;
      void* biasDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      void* meanDeviceAddr = nullptr;
      void* rstdDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclIntArray* norm = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* bias = nullptr;
      aclTensor* out = nullptr;
      aclTensor* mean = nullptr;
      aclTensor* rstd = nullptr;
      std::vector<float> xHostData(64, 2.0);
      std::vector<int64_t> normData = {32};
      std::vector<float> weightHostData(32, 1.0);
      std::vector<float> biasHostData(32, 0.0);
      std::vector<float> outHostData(64, 0.0);
      std::vector<float> meanHostData(2, 0.0);
      std::vector<float> rstdHostData(2, 0.0);
      double eps = 1e-5;

      ret = CreateAclTensorMem(xHostData, xShape, &xDeviceAddr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensorMem(weightHostData, normShape, &weightDeviceAddr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensorMem(biasHostData, normShape, &biasDeviceAddr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensorMem(outHostData, xShape, &outDeviceAddr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensorMem(meanHostData, meanShape, &meanDeviceAddr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensorMem(rstdHostData, meanShape, &rstdDeviceAddr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      aclCreateTensorP(xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
      aclCreateIntArrayP(normData, &norm);
      aclCreateTensorP(normShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
      aclCreateTensorP(normShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
      aclCreateTensorP(xShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
      aclCreateTensorP(meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
      aclCreateTensorP(meanShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);

      // 3. Call the CANN operator library API, and replace it with the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      // Call the first-phase API of aclnnLayerNorm.
      ret = aclnnLayerNormGetWorkspaceSize(x, norm, weight, bias, eps, out, mean, rstd, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnLayerNorm.
      ret = aclnnLayerNorm(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNorm failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronously wait for task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value, copy the result from the device memory to the host side, and modify it based on the specific API definition.
      auto size = GetShapeSize(xShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy first result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("out result[%ld] is: %f\n", i, resultData[i]);
      }

      auto size1 = GetShapeSize(meanShape);
      std::vector<float> resultData1(size1, 0);
      ret = aclrtMemcpy(
          resultData1.data(), resultData1.size() * sizeof(resultData1[0]), meanDeviceAddr, size1 * sizeof(resultData1[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy second result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size1; i++) {
          LOG_PRINT("mean result[%ld] is: %f\n", i, resultData1[i]);
      }

      auto size2 = GetShapeSize(meanShape);
      std::vector<float> resultData2(size2, 0);
      ret = aclrtMemcpy(
          resultData2.data(), resultData2.size() * sizeof(resultData2[0]), rstdDeviceAddr, size2 * sizeof(resultData2[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy last result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size2; i++) {
          LOG_PRINT("rstd result[%ld] is: %f\n", i, resultData2[i]);
      }

      // 6. Release aclTensor and aclIntArray based on the specific API definition.
      aclDestroyTensor(x);
      aclDestroyIntArray(norm);
      aclDestroyTensor(weight);
      aclDestroyTensor(bias);
      aclDestroyTensor(out);
      aclDestroyTensor(mean);
      aclDestroyTensor(rstd);

      // 7. Release device resources.
      aclrtFree(xDeviceAddr);
      aclrtFree(weightDeviceAddr);
      aclrtFree(biasDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclrtFree(meanDeviceAddr);
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

- aclnnLayerNormWithImplMode example:

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_layer_norm.h"

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
      // Boilerplate, initialize resources.
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
  int CreateAclTensorMem(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy data from the host to the device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  void aclCreateTensorP(const std::vector<T>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
  {
      // Calculate the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create aclTensor.
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
  }

  template <typename T>
  void aclCreateIntArrayP(const std::vector<T>& hostData, aclIntArray** intArray)
  {
      // Call the API to create aclIntArray.
      *intArray = aclCreateIntArray(hostData.data(), hostData.size());
  }

  int main()
  {
      // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
      // Set deviceId according to the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Customize the construction according to the API.
      std::vector<int64_t> xShape = {1, 2, 32};
      std::vector<int64_t> normShape = {32};
      std::vector<int64_t> meanShape = {1, 2, 1};
      void* xDeviceAddr = nullptr;
      void* weightDeviceAddr = nullptr;
      void* biasDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      void* meanDeviceAddr = nullptr;
      void* rstdDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclIntArray* norm = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* bias = nullptr;
      aclTensor* out = nullptr;
      aclTensor* mean = nullptr;
      aclTensor* rstd = nullptr;
      std::vector<float> xHostData(64, 2.0);
      std::vector<int64_t> normData = {32};
      std::vector<float> weightHostData(32, 1.0);
      std::vector<float> biasHostData(32, 0.0);
      std::vector<float> outHostData(64, 0.0);
      std::vector<float> meanHostData(2, 0.0);
      std::vector<float> rstdHostData(2, 0.0);
      double eps = 1e-5;
      int32_t implMode = 0;

      // Create the x aclTensor.
      ret = CreateAclTensorMem(xHostData, xShape, &xDeviceAddr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the weight aclTensor.
      ret = CreateAclTensorMem(weightHostData, normShape, &weightDeviceAddr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create bias aclTensor.
      ret = CreateAclTensorMem(biasHostData, normShape, &biasDeviceAddr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create out aclTensor.
      ret = CreateAclTensorMem(outHostData, xShape, &outDeviceAddr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create mean aclTensor.
      ret = CreateAclTensorMem(meanHostData, meanShape, &meanDeviceAddr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create rstd aclTensor.
      ret = CreateAclTensorMem(rstdHostData, meanShape, &rstdDeviceAddr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      aclCreateTensorP(xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
      aclCreateIntArrayP(normData, &norm);
      aclCreateTensorP(normShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
      aclCreateTensorP(normShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
      aclCreateTensorP(xShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
      aclCreateTensorP(meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
      aclCreateTensorP(meanShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);

      // 3. Call a CANN operator library API. Change it to the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      // Call the aclnnLayerNormWithImplMode first-phase API.
      ret = aclnnLayerNormWithImplModeGetWorkspaceSize(
          x, norm, weight, bias, eps, out, mean, rstd, implMode, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormWithImplModeGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the aclnnLayerNormWithImplMode second-phase API.
      ret = aclnnLayerNormWithImplMode(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormWithImplMode failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output values, copy the result from the device side to the host side, and modify it based on the specific API.
      auto size = GetShapeSize(xShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy first result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("out result[%ld] is: %f\n", i, resultData[i]);
      }

      auto size1 = GetShapeSize(meanShape);
      std::vector<float> resultData1(size1, 0);
      ret = aclrtMemcpy(
          resultData1.data(), resultData1.size() * sizeof(resultData1[0]), meanDeviceAddr, size1 * sizeof(resultData1[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy second result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size1; i++) {
          LOG_PRINT("mean result[%ld] is: %f\n", i, resultData1[i]);
      }

      auto size2 = GetShapeSize(meanShape);
      std::vector<float> resultData2(size2, 0);
      ret = aclrtMemcpy(
          resultData2.data(), resultData2.size() * sizeof(resultData2[0]), rstdDeviceAddr, size2 * sizeof(resultData2[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy last result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size2; i++) {
          LOG_PRINT("rstd result[%ld] is: %f\n", i, resultData2[i]);
      }

      // 6. Release aclTensor and aclIntArray, modifying as required by the API definition of the specific API.
      aclDestroyTensor(x);
      aclDestroyIntArray(norm);
      aclDestroyTensor(weight);
      aclDestroyTensor(bias);
      aclDestroyTensor(out);
      aclDestroyTensor(mean);
      aclDestroyTensor(rstd);

      // 7. Release device resources.
      aclrtFree(xDeviceAddr);
      aclrtFree(weightDeviceAddr);
      aclrtFree(biasDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclrtFree(meanDeviceAddr);
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
