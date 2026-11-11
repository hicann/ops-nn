# aclnnAddRmsNormQuantV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:53:34.842Z -->

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: The RmsNorm operator is a normalization operation commonly used in large models. Compared with the LayerNorm operator, it removes the mean subtraction part. The AddRmsNormQuant operator fuses the Add operator before RmsNorm with the Quantize operator after RmsNorm, reducing data movement operations. Compared with AddRmsNormQuant, the AddRmsNormQuantV2 operator adds the bias term `betaOptional` parameter and the RmsNorm normalization result output `rmsNormOut`.

- Formula:

  $$
  x_i={x1}_i+{x2}_i
  $$

  $$
  y_i=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * g_i + beta_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  $$
  rmsNormOut_i=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * g_i
  $$

  - When divMode is true:

    $$
    y1Out=round((y/scales1)+zero\_points1)
    $$

    $$
    y2Out=round((y/scales2)+zero\_points2)
    $$

  - When divMode is false:

    $$
    y1Out=round((y*scales1)+zero\_points1)
    $$

    $$
    y2Out=round((y*scales2)+zero\_points2)
    $$

  > **NOTE**: `rmsNormOut` is the RmsNorm normalization result (excluding beta). `betaOptional` and `rmsNormOut` are independent of each other and can be used in any combination.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnAddRmsNormQuantV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the computation process. Then call the "aclnnAddRmsNormQuantV2" API to perform the computation.

```Cpp
aclnnStatus aclnnAddRmsNormQuantV2GetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  const aclTensor *scales1,
  const aclTensor *scales2Optional,
  const aclTensor *zeroPoints1Optional,
  const aclTensor *zeroPoints2Optional,
  const aclTensor *betaOptional,
  int64_t          axis,
  double           epsilon,
  bool             divMode,
  aclTensor       *y1Out,
  aclTensor       *y2Out,
  aclTensor       *xOut,
  aclTensor       *rmsNormOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddRmsNormQuantV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddRmsNormQuantV2GetWorkspaceSize

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
      <td>x1 (const aclTensor*)</td>
      <td>Input</td>
      <td>Source data tensor in the normalization process. Corresponds to `x1` in the formula.</td>
      <td>Supports null Tensor.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2 (const aclTensor*)</td>
      <td>Input</td>
      <td>Source data tensor in the normalization process. Corresponds to `x2` in the formula.</td>
      <td><ul><li>Supports null tensor.</li><li>shape and data type remain consistent with `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (const aclTensor*)</td>
      <td>Input</td>
      <td>Weight tensor in the normalization. Corresponds to `g` in the formula.</td>
      <td><ul><li>Supports null tensor.</li><li>shape remains consistent with the dimensions of `x1` that require norm (layer normalization), and the data type remains consistent with `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales1 (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the scales tensor used to obtain y1Out in quantization, corresponding to `scales1` in the formula.</td>
      <td><ul><li>Supports null Tensor.</li><li>The shape remains consistent with `gamma`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales2Optional (const aclTensor*)</td>
      <td>Input</td>
      <td>Scales tensor used to obtain y2Out in quantization. Corresponds to `scales2` in the formula.</td>
      <td><ul><li>Supports null tensors.</li><li>Optional parameter, supports passing in a null pointer. Shape and data type remain consistent with `scales1`. For combined constraints with `zeroPoints2Optional`, see <a href="#constraints">Constraints</a>.</li><li>When `divMode` is set to True, the value of this parameter cannot be 0.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints1Optional (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the offset tensor used to obtain y1Out during quantization. Corresponds to `zero_points1` in the formula.</td>
      <td><ul><li>Supports null Tensor.</li><li>Optional parameter, supports passing in a null pointer. shape remains consistent with `scales1`.</li></ul></td>
      <td>INT32, FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints2Optional (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the offset tensor used to obtain y2Out during quantization. Corresponds to `zero_points2` in the formula.</td>
      <td><ul><li>Null tensors are supported.</li><li>Optional parameter. Null pointers are supported. The shape must remain consistent with `scales1`, and the data type must remain consistent with `zeroPoints1Optional`. For combination constraints with `scales2Optional`, see <a href="#constraints">Constraints</a>.</li></ul></td>
      <td>INT32, FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>betaOptional (const aclTensor*)</td>
      <td>Input</td>
      <td>Bias term in normalization. Corresponds to `beta` in the formula.</td>
      <td><ul><li>Null tensors are supported.</li><li>Optional parameter. Null pointers are supported.</li><li>The shape and data type must remain consistent with `gamma`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis (int64_t)</td>
      <td>Input</td>
      <td>Indicates the elewise axis to quantize. Other axes are broadcast. The specified axis cannot exceed the number of dimensions of input `x1`. Currently only -1 is supported. Other values have no effect.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon (double)</td>
      <td>Input</td>
      <td>eps in the formula, used to prevent division-by-zero errors. The data type is double. A small positive number is recommended.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divMode (bool)</td>
      <td>Input</td>
      <td>Parameter that determines whether the quantization formula uses division. The data type is bool.</td>
      <td>Supports True and False.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantization output tensor, corresponding to `y1Out` in the formula.</td>
      <td><ul><li>Supports null tensor.</li><li>The shape is consistent with that of input `x1`/`x2`.</li></ul></td>
      <td>INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantization output tensor, corresponding to `y2Out` in the formula.</td>
      <td><ul><li>Supports null tensor.</li><li>The shape and data type remain consistent with those of `y1Out`.</li></ul></td>
      <td>INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut(aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the sum of x1 and x2, corresponding to `x` in the formula.</td>
      <td><ul><li>Supports empty Tensor.</li><li>The shape and data type remain consistent with `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rmsNormOut (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the result after RmsNorm (excluding beta), corresponding to `rmsNormOut` in the corresponding formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Optional output. Supports passing in a null pointer. When a null pointer is passed in, rmsNormOut is not output.</li><li>The shape and data type remain consistent with `x1`.</li></ul></td>
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
      <td>executor(aclOpExecutor**)</td>
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

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>:
    - Data type:
      - Input parameters `x1`, `x2`, and `gamma`, and output parameters `xOut` and `rmsNormOut` only support FLOAT16.
      - The input parameters `scales1` and `scales2Optional` only support FLOAT32.
      - The optional parameters `zeroPoints1Optional` and `zeroPoints2Optional` only support INT32.
      - Optional parameter `betaOptional` only supports FLOAT16.
      - The output parameters `y1Out` and `y2Out` only support INT8.
    - Input parameter `divMode` on the native V2 path only supports True. When `xOut` is non-null, `rmsNormOut` is null, and other parameters meet the V1 fallback path constraints, passing in False is supported. The API falls back to the V1 path and runs in multiplication mode.
  <!-- end id7 -->

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - Data types:
      - The input parameters `x1`, `x2`, and `gamma` and output parameters `xOut` and `rmsNormOut` only support FLOAT16 and BFLOAT16.
      - The input parameters `scales1` and `scales2Optional` only support FLOAT32 and BFLOAT16.
      - The optional parameters `zeroPoints1Optional` and `zeroPoints2Optional` only support INT32 and BFLOAT16.
      - The optional parameter `betaOptional` only supports FLOAT16 and BFLOAT16.
      - The output parameters `y1Out` and `y2Out` only support INT8.
    - The input parameter `divMode` only supports True in the native V2 path. When `xOut` is non-null, `rmsNormOut` is null, and other parameters meet the V1 fallback path constraints, the API supports passing in False. It falls back to the V1 path and executes in multiplication mode.
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
      <td>If a passed-in parameter is a required input, output, or required attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of the input or output is not within the supported range.
      </td>
    </tr>
  </tbody></table>

## aclnnAddRmsNormQuantV2

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
      <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnAddRmsNormQuantV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- The size of each dimension in the shape of parameters `x1`, `x2`, `gamma`, `scales1`, `scales2Optional`, `zeroPoints1Optional`, `zeroPoints2Optional`, `betaOptional`, `y1Out`, `y2Out`, `xOut`, and `rmsNormOut` must not exceed the maximum INT32 value of 2147483647.

<!-- npu="950" id13 -->
- <term>Ascend 950PR&950DT products</term>: Supports passing in a null pointer for `scales2Optional` with `zeroPoints2Optional` non-null. In this case, `scales2` of the second quantization is processed as 1. That is, $y2Out=round(y+zero\_points2)$ holds regardless of whether `divMode` is True or False, and the shape of `y2Out` remains consistent with that of `x1`.
<!-- end id13 -->

<!-- npu="A3,910b,310p" id14 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, and <term>Atlas inference products</term>: Do not support passing in only `zeroPoints2Optional` without `scales2Optional`. If `zeroPoints2Optional` is non-null, `scales2Optional` must also be non-null.
<!-- end id14 -->

<!-- npu="310p" id9 -->
- <term>Atlas inference products</term>: The norm axis length of `x1`, `x2`, `y1Out`, `y2Out`, `xOut`, and `rmsNormOut`, and the length of `gamma`, `scales1`, `scales2Optional`, `zeroPoints1Optional`, `zeroPoints2Optional`, and `betaOptional` must be greater than or equal to 32 Bytes.
<!-- end id9 -->

- `betaOptional` and `rmsNormOut` are independent of each other:
  - When `rmsNormOut` is output, `betaOptional` can be present or absent.
  - When `rmsNormOut` is not output (a null pointer is passed in), `betaOptional` can still be present or absent.
  - The computation result of `rmsNormOut` is $\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * g_i$, excluding the beta computation.

- Data format description:

  It is recommended to use the ND format for all input and output tensors. Other data formats are converted to the ND format by the framework by default for processing.

- Supported data types for each product model:

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    | x1 Data Type | x2 Data Type | Gamma Data Type | scales1 Data Type | scales2Optional Data Type | zeroPoints1Optional Data Type | zeroPoints2Optional Data Type | betaOptional Data Type | y1Out Data Type | y2Out Data Type | xOut Data Type | rmsNormOut Data Type |
    | - | - | - | - | - | - | - | - | - | - | - | - |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 | FLOAT16 | INT8 | INT8 | FLOAT16 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | INT8 | INT8 | BFLOAT16 | BFLOAT16 |
  <!-- end id10 -->

  <!-- npu="310p" id11 -->
  - <term>Atlas inference products</term>:

    | x1 Data Type | x2 Data Type | Gamma Data Type | scales1 Data Type | scales2Optional Data Type | zeroPoints1Optional Data Type | zeroPoints2Optional Data Type | betaOptional Data Type | y1Out Data Type | y2Out Data Type | xOut Data Type | rmsNormOut Data Type |
    | - | - | - | - | - | - | - | - | - | - | - | - |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 | FLOAT16 | INT8 | INT8 | FLOAT16 | FLOAT16 |
  <!-- end id11 -->

  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>:

    | x1 Data Type | x2 Data Type | Gamma Data Type | scales1 Data Type | scales2Optional Data Type | zeroPoints1Optional Data Type | zeroPoints2Optional Data Type | betaOptional Data Type | y1Out Data Type | y2Out Data Type | xOut Data Type | rmsNormOut Data Type |
    | - | - | - | - | - | - | - | - | - | - | - | - |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 | FLOAT16 | INT8 | INT8 | FLOAT16 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 | BFLOAT16 | INT8 | INT8 | BFLOAT16 | BFLOAT16 |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | INT8 | INT8 | FLOAT32 | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | INT8 | INT8 | FLOAT16 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | INT8 | INT8 | BFLOAT16 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT16 | INT8 | INT8 | FLOAT16 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | BFLOAT16 | INT8 | INT8 | BFLOAT16 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 | FLOAT16 | HIFLOAT8 | HIFLOAT8 | FLOAT16 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 | BFLOAT16 | HIFLOAT8 | HIFLOAT8 | BFLOAT16 | BFLOAT16 |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | HIFLOAT8 | HIFLOAT8 | FLOAT16 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | HIFLOAT8 | HIFLOAT8 | BFLOAT16 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT16 | HIFLOAT8 | HIFLOAT8 | FLOAT16 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | BFLOAT16 | HIFLOAT8 | HIFLOAT8 | BFLOAT16 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 | FLOAT16 | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT16 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 | BFLOAT16 | FLOAT8_E5M2 | FLOAT8_E5M2 | BFLOAT16 | BFLOAT16 |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT16 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT8_E5M2 | FLOAT8_E5M2 | BFLOAT16 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT16 | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT16 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | BFLOAT16 | FLOAT8_E5M2 | FLOAT8_E5M2 | BFLOAT16 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 | FLOAT16 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT16 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 | BFLOAT16 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | BFLOAT16 | BFLOAT16 |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT16 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | BFLOAT16 | BFLOAT16 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT16 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT16 | FLOAT16 |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | BFLOAT16 | FLOAT8_E4M3FN | FLOAT8_E4M3FN | BFLOAT16 | BFLOAT16 |

    The data types of `x1`, `x2`, `gamma`, non-null `betaOptional`, `xOut`, and non-null `rmsNormOut` must be the same; the data types of `scales1` and non-null `scales2Optional` must be the same; when `zeroPoints1Optional` and `zeroPoints2Optional` are both non-null, their data types must be the same; the data types of `y1Out` and `y2Out` must be the same. The data type combinations of quantization parameters must follow those in the table.
  <!-- end id12 -->

- Deterministic computation:
  - aclnnAddRmsNormQuantV2 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm_quant_v2.h"

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
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs. Customize the construction based on the API.
    std::vector<int64_t> xShape = {64, 32};
    std::vector<int64_t> gammaShape = {32};
    std::vector<int64_t> yShape = {64, 32};
    long long xShapeSize = GetShapeSize(xShape);
    long long gammaShapeSize = GetShapeSize(gammaShape);
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* scales1DeviceAddr = nullptr;
    void* zeroPoints1DeviceAddr = nullptr;
    void* y1DeviceAddr = nullptr;
    void* y2DeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* rmsNormOutDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* scales1 = nullptr;
    aclTensor* zeroPoints1 = nullptr;
    aclTensor* y1 = nullptr;
    aclTensor* y2 = nullptr;
    aclTensor* x = nullptr;
    aclTensor* rmsNormOut = nullptr;
    std::vector<int16_t> x1HostData(xShapeSize, 0);
    std::vector<int16_t> x2HostData(xShapeSize, 0);
    std::vector<int16_t> gammaHostData(gammaShapeSize, 0);
    std::vector<int16_t> betaHostData(gammaShapeSize, 0);
    std::vector<float> scales1HostData(gammaShapeSize, 1);
    std::vector<int32_t> zeroPoints1HostData(gammaShapeSize, 100);
    std::vector<int8_t> y1HostData(xShapeSize, 0);
    std::vector<int8_t> y2HostData(xShapeSize, 0);
    std::vector<int16_t> xHostData(xShapeSize, 0);
    std::vector<int16_t> rmsNormOutHostData(xShapeSize, 0);
    float epsilon = 1e-6;
    int64_t axis = -1;
    bool divMode = true;
    // Create x1 aclTensor
    ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create x2 aclTensor
    ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create gamma aclTensor
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create beta aclTensor
    ret = CreateAclTensor(betaHostData, gammaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create scales1 aclTensor
    ret = CreateAclTensor(scales1HostData, gammaShape, &scales1DeviceAddr, aclDataType::ACL_FLOAT, &scales1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create zeroPoints1 aclTensor
    ret =
        CreateAclTensor(zeroPoints1HostData, gammaShape, &zeroPoints1DeviceAddr, aclDataType::ACL_INT32, &zeroPoints1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create y1 aclTensor
    ret = CreateAclTensor(y1HostData, yShape, &y1DeviceAddr, aclDataType::ACL_INT8, &y1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create y2 aclTensor
    ret = CreateAclTensor(y2HostData, yShape, &y2DeviceAddr, aclDataType::ACL_INT8, &y2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the rmsNormOut aclTensor.
    ret = CreateAclTensor(rmsNormOutHostData, xShape, &rmsNormOutDeviceAddr, aclDataType::ACL_FLOAT16, &rmsNormOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAddRmsNormQuantV2.
    ret = aclnnAddRmsNormQuantV2GetWorkspaceSize(
        x1, x2, gamma, scales1, nullptr, zeroPoints1, nullptr, beta, axis, epsilon, divMode, y1, y2, x, rmsNormOut,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory using the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnAddRmsNormQuantV2.
    ret = aclnnAddRmsNormQuantV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormQuantV2 failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from device memory to the host side. Modify the code based on the specific API definition.
    auto size = GetShapeSize(yShape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), y1DeviceAddr, size * sizeof(int8_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(scales1);
    aclDestroyTensor(zeroPoints1);
    aclDestroyTensor(y1);
    aclDestroyTensor(y2);
    aclDestroyTensor(x);
    aclDestroyTensor(rmsNormOut);

    // 7. Release device resources. Modify the code based on the specific API definition.
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(scales1DeviceAddr);
    aclrtFree(zeroPoints1DeviceAddr);
    aclrtFree(y1DeviceAddr);
    aclrtFree(y2DeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(rmsNormOutDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
