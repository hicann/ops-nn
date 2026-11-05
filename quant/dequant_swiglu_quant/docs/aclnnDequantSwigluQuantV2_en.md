# aclnnDequantSwigluQuantV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:12:24.979Z -->

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

- Description: Adds dequant and quant operations before and after the Swish gated linear unit (SwiGLU) activation function to perform DequantSwigluQuant computation on x. Compared with [aclnnDequantSwigluQuant](aclnnDequantSwigluQuant.md), this API adds two new parameter types:

  <!-- npu="950" id9 -->
  1) Three input parameters are added for <term>Ascend 950PR&950DT products</term>: dstType, roundModeOptional, and activateDim.
  <!-- end id9 -->

  <!-- npu="950,A3,910b" id8 -->
  2) On <term>Atlas A3 products</term> and <term>Atlas A2 products</term>, four parameters are added for the GPT-OSS variant SwiGLU: swigluMode, clampLimit, gluAlpha, and gluBias; when this API is called on <term>Ascend 950PR&950DT products</term>, set these four parameters to their default values. Choose an appropriate API based on the actual requirements.
  <!-- end id8 -->

- Formula when swigluMode is 0:

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  swigluOut_i = Swiglu(dequantOut_i)=Swish(A_i)*B_i
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  Where A<sub>i</sub> represents the first half of dequantOut<sub>i</sub>, and B<sub>i</sub> represents the second half of dequantOut<sub>i</sub>.

- Formula when swigluMode is 1:

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  x\_glu = x\_glu.clamp(min=None, max=clampLimit)
  $$

  $$
  x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)
  $$

  $$
  out\_glu = x\_glu * sigmoid(gluAlpha * x\_glu)
  $$

  $$
  swigluOut_i = out\_glu * (x\_linear + gluBias)
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  where x_glu represents the even-indexed part of dequantOut<sub>i</sub>, and x_linear represents the odd-indexed part of dequantOut<sub>i</sub>.

- Formula when swigluMode is 2:

  The computation logic is the same as that when swigluMode is 1 (both are the GPT-OSS variant SwiGLU, using clampLimit, gluAlpha, and gluBias):

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  x\_glu = x\_glu.clamp(min=None, max=clampLimit)
  $$

  $$
  x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)
  $$

  $$
  out\_glu = x\_glu * sigmoid(gluAlpha * x\_glu)
  $$

  $$
  swigluOut_i = out\_glu * (x\_linear + gluBias)
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  The difference from swigluMode 1 lies in the splitting method of x\_glu and x\_linear: when swigluMode is 2, x\_glu denotes the first half of dequantOut<sub>i</sub>, and x\_linear denotes the second half of dequantOut<sub>i</sub> (the same splitting method as that of swigluMode 0); when swigluMode is 1, the values are split by interleaving odd and even indexes.

- Formula when swigluMode is 3 (variant SwiGLU, using clampLimit):

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  gate = dequantOut_i[:, :H/2]
  $$

  $$
  up = dequantOut_i[:, H/2:]
  $$

  $$
  gate = SiLU(gate) = gate * sigmoid(gate)
  $$

  $$
  gate = gate.clamp(max=clampLimit)
  $$

  $$
  up = up.clamp(min=-clampLimit, max=clampLimit)
  $$

  $$
  swigluOut_i = gate * up
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  where gate indicates the first half of dequantOut<sub>i</sub>, and up indicates the second half of dequantOut<sub>i</sub> (the same splitting method as swigluMode 0). swigluMode 3 combines the SiLU activation of swigluMode 0 and the front-back splitting and clamping mechanism of swigluMode 2, but does not use the gluAlpha and gluBias parameters.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnDequantSwigluQuantV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnDequantSwigluQuantV2" API to perform computation.

```Cpp
aclnnStatus aclnnDequantSwigluQuantV2GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weightScaleOptional,
  const aclTensor *activationScaleOptional,
  const aclTensor *biasOptional,
  const aclTensor *quantScaleOptional,
  const aclTensor *quantOffsetOptional,
  const aclTensor *groupIndexOptional,
  bool             activateLeft,
  char            *quantModeOptional,
  int64_t          dstType,
  char            *roundModeOptional,
  int64_t          activateDim,
  int64_t          swigluMode,
  double           clampLimit,
  double           gluAlpha,
  double           gluBias,
  const aclTensor *yOut,
  const aclTensor *scaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDequantSwigluQuantV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDequantSwigluQuantV2GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
  <col style="width: 301px">
  <col style="width: 115px">
  <col style="width: 150px">
  <col style="width: 350px">
  <col style="width: 177px">
  <col style="width: 104px">
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
      <th>Non-contiguous Tensor</th>
    </tr></thead>
   <tbody>
       <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Data to be processed, which is x in the formula.</td>
      <td><ul><li>The shape is [X1,X2,...Xn,2H], with no fewer than 2 dimensions and no more than 8 dimensions.</li><li>The dimension of input x corresponding to activateDim must be a multiple of 2.</li></ul></td>
      <td>FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>x</td>
    </tr>
     <tr>
      <td>weightScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Dequantization scale of weight.</td>
      <td><ul><li>The shape is 1D or 2D, expressed as [2H] or [groupNum, 2H], and 2H must be consistent with the last dimension of x.</li><li>Optional parameter, supports passing a null pointer. When groupIndexOptional is a null pointer, the shape is [2H] or [1, 2H]; when groupIndexOptional is not a null pointer, the shape is [groupNum, 2H].</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>x</td>
      </tr>
    <tr>
      <td>activationScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Dequant scale of the activation function.</td>
      <td><ul><li>Dequant scale of the activation function.</li><li>The shape is [X1,X2,...Xn], with 1 to 7 dimensions. The number of dimensions is one less than that of x, and the shape must be consistent with the shape of x in corresponding dimensions.</li><li>Optional parameter, supports passing a null pointer.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-7</td>
      <td>x</td>
    </tr>
      <tr>
      <td>biasOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Bias of Matmul, corresponding to biasOptional in the formula.</td>
      <td><ul><li>The shape supports 1 or 2 dimensions, represented as [2H] or [groupNum, 2H]. The value of 2H must be consistent with the last dimension of x. If groupIndexOptional is a null pointer, the shape is [2H]; if groupIndexOptional is not a null pointer, the shape is [groupNum, 2H].</li><li>Optional parameter. A null pointer is supported.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>x</td>
    </tr>
       <tr>
      <td>quantScaleOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Scale for quantization, corresponding to quantScaleOptional in the formula.</td>
      <td><ul><li>Optional parameter, supports passing a null pointer. It is mandatory when quantModeOptional is static.</li><li>The shape is 1D or 2D. For details about the constraints, see the product-specific description below.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>x</td>
    </tr>
       <tr>
      <td>quantOffsetOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Quantization offset.</td>
      <td><ul><li>Optional parameter, supports passing a null pointer. It is supported only when quantModeOptional is static; passing a non-null pointer in dynamic mode causes an error.</li><li>The shape is 1D or 2D. For details about the constraints, see the product-specific description below.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>x</td>
    </tr>
      <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input</td>
      <td>group_index required for MoE grouping.</td>
      <td><ul><li>Supports 1D or 2D tensors. The shape is [groupNum] or [groupNum, 2], and groupNum must be greater than or equal to 1.</li><li>Optional parameter. An empty pointer is supported.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>x</td>
    </tr>
      <tr>
      <td>activateLeft (bool)</td>
      <td>Input</td>
      <td>Indicates whether to apply the swiglu activation to the left half of the input.</td>
      <td>If the value is false, the activation is applied to the right half of the input. If swigluMode is 1, activateLeft must be true.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>quantModeOptional (char*)</td>
      <td>Input</td>
      <td>Specifies whether to use dynamic or static quantization.</td>
      <td><ul><li>Supports "dynamic" and "static".</li><li>Supports null pointer input. If a null pointer is passed in, "static" is used by default.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Specifies the data type of output y.</td>
      <td><ul><li>The value range of dstType is [2, 34,35, 36, 40, 41], corresponding to INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, and FLOAT4_E1M2 respectively.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>roundModeOptional (char*)</td>
      <td>Input</td>
      <td>Rounding mode for output y.</td>
      <td><ul><li>Value range: ["rint", "round", "floor", "ceil", "trunc"].</li><li>If the data type of output y is INT8, FLOAT8_E5M2, or FLOAT8_E4M3FN, only the "rint" mode is supported.</li><li>If the data type of output y is HIFLOAT8, only the "round" mode is supported.</li><li>A null pointer is allowed. If a null pointer is passed in, "rint" is used by default.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>activateDim (int64_t)</td>
      <td>Input</td>
      <td>Specifies the axis for the swish calculation.</td>
      <td><ul><li>The value range of activateDim is [-xDim, xDim - 1] (where xDim is the dimension of input x).</li><li>If activateDim does not correspond to the trailing axis of x, groupIndexOptional cannot be provided.</li><li>If activateDim does not correspond to the trailing axis of x, quantModeOptional supports only dynamic.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>swigluMode (int64_t)</td>
      <td>Input</td>
      <td>Specifies the computation mode of swiglu.</td>
      <td><ul><li>Value range: [0, 1, 2, 3]. Default value: 0.</li><li>0: traditional swiglu computation.</li><li>1: swiglu variant (GPT-OSS variant). It uses odd-even partitioning and supports clampLimit, the gluAlpha activation coefficient, and the gluBias bias.</li><li>2: The computation is the same as that of 1 (both are swiglu variants and support clampLimit, gluAlpha, and gluBias). The difference is the partitioning: contiguous front-back-half partitioning (the same as 0) instead of odd-even partitioning.</li><li>3: swiglu variant. The partitioning is the same as 0 and 2 (contiguous front-back-half partitioning). The computation differs from 2 in that clamp is applied after silu. gluAlpha and gluBias are not supported.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>clampLimit (double)</td>
      <td>Input</td>
      <td>Indicates the threshold value used by the swiglu variant.</td>
      <td><ul><li>This parameter is optional.</li><li>Used to clamp the input. The value must be greater than 0 and less than infinity. This prevents excessively large values from affecting the stability of swiglu computation. The default value is 7.0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>gluAlpha(double)</td>
      <td>Input</td>
      <td>Specifies the parameter used by the swiglu variant.</td>
      <td><ul><li>This parameter is optional.</li><li>Used to adjust the scaling of the linear part in the glu activation function. The default value is 1.702.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>gluBias(double)</td>
      <td>Input</td>
      <td>Bias parameter used by the swiglu variant.</td>
      <td><ul><li>This parameter is optional.</li><li>Adds an offset to the linear computation in swiglu. The default value is 1.0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>-</td>
      <td><ul><li>When activateDim corresponds to the tail axis of x, the shape is [X1,X2,...Xn,H].</li><li>When activateDim does not correspond to the tail axis of x, the shape is [X1,X2,...,XactivateDim / 2,...,2H].</li><li>When the data type of yOut is FLOAT4_E2M1 or FLOAT4_E1M2, the last dimension of yOut must be a multiple of 2.</li><li>When activateDim does not correspond to the tail axis of x, the tail axis of yOut must be less than 5120.</li></ul></td>
      <td>INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td>ND</td>
      <td>2-8</td>
      <td>x</td>
    </tr>
    <tr>
      <td>scaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>-</td>
      <td><ul><li>When activateDim corresponds to the tail axis of x, the shape is [X1,X2,...,Xn].</li><li>When activateDim does not correspond to the tail axis of x, the shape is [X1,X2,...,XactivateDim / 2,...,Xn].</li><li>When quantModeOptional is static, scaleOut is not computed.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-7</td>
      <td>x</td>
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
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - For x, only 2-dimensional inputs with data type INT32 or BFLOAT16 are supported.
    - quantScaleOptional parameter: the shape validation rules depend on whether groupIndexOptional is a null pointer and on the value of quantModeOptional (H is half of the last dimension of x, that is, the last dimension of y):
      - When groupIndexOptional is a null pointer (in this case quantModeOptional supports only dynamic), a null pointer is supported; if not null, the shape is [H] or [1, H].
      - When groupIndexOptional is not a null pointer (groupNum is the first dimension of groupIndexOptional):
        - static: mandatory. The shape is [groupNum] (one quant parameter per group) or [groupNum, H] (per channel per group). quantOffsetOptional must follow the same shape rules.
        - dynamic: optional. If not null, the shape is [H] or [1, H] when groupNum is 1, or [groupNum, H] when groupNum is greater than 1 (the first dimension must be groupNum).
    - quantOffsetOptional parameter: its constraints also depend on whether groupIndexOptional is a null pointer and on the value of quantModeOptional (H is half of the last dimension of x, that is, the last dimension of y):
      - When groupIndexOptional is a null pointer, it must be a null pointer.
      - When groupIndexOptional is not a null pointer (groupNum is the first dimension of groupIndexOptional):
        - static: mandatory, and the data type and shape must be the same as those of quantScaleOptional.
        - dynamic: must be a null pointer.
    - groupIndexOptional parameter: only INT64 is supported. This parameter is optional and a null pointer is supported. The shape can be 1-D or 2-D: 1-D is [groupNum], where each element indicates the number of tokens in the group; 2-D is [groupNum, 2], where each row indicates [realGroupIdx, count]. When x is BFLOAT16, only the 1-D shape is supported; the 2-D shape cannot be used together with biasOptional or a non-zero swigluMode.
    - biasOptional parameter: when groupIndexOptional is a null pointer and swigluMode is not 0, biasOptional is not supported and a null pointer must be passed.
    - dstType parameter: only 2-INT8 input is supported.
    - roundModeOptional parameter: only a null pointer or "rint" is supported.
    - activateDim parameter: only -1 is supported.
  <!-- end id7 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>:
    - In the quantScaleOptional parameter, the data type must be FLOAT and the shape must be 1D or 2D. The parameter is mandatory when quantModeOptional is static and supports a null pointer in dynamic mode (H is the last dimension of y):
      - When groupIndexOptional is a null pointer, the shape is [1] or [H] in static mode and [H] or [1, H] in dynamic mode.
      - When groupIndexOptional is not a null pointer (groupNum is the first dimension of groupIndexOptional), the first dimension must be groupNum in static mode and the shape is [groupNum] or [groupNum, H] (the last dimension is H or 1); in dynamic mode, the shape must be [groupNum, H].
    - In the quantOffsetOptional parameter, the data type must be FLOAT and the shape must be 1D or 2D. The parameter is optional and is supported only when quantModeOptional is static; passing a non-null pointer in dynamic mode causes an error:
      - When groupIndexOptional is a null pointer, the shape is [1] or [H].
      - When groupIndexOptional is not a null pointer (groupNum is the first dimension of groupIndexOptional), the first dimension must be groupNum and the shape is [groupNum] or [groupNum, H] (the last dimension is H or 1).
      - When both quantScaleOptional and quantOffsetOptional are provided, they must have the same number of elements.
  <!-- end id11 -->

- **Return value**

  aclnnStatus: returned status code. For details, see [aclnn return codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>Required parameter x, yOut, or scaleOut is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of the input or output is not in the supported range.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="7">561002</td>
      <td>weightScaleOptional is a null pointer when the data type of x is int32.</td>
    </tr>
    <tr>
      <td>quantScaleOptional is a null pointer when quantModeOptional is static.</td>
    </tr>
    <tr>
      <td>quantModeOptional is out of the specified value range.</td>
    </tr>
    <tr>
      <td>dstType is not in the specified value range.</td>
    </tr>
    <tr>
      <td>roundModeOptional is not in the specified value range.</td>
    </tr>
    <tr>
      <td>activateDim is not in the specified value range.</td>
    </tr>
    <tr>
      <td>The shapes of weightScaleOptional, activationScaleOptional, biasOptional, quantScaleOptional, and groupIndexOptional do not meet the constraints with x.</td>
    </tr>
  </tbody></table>

## aclnnDequantSwigluQuantV2

- **Parameters:**
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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by calling the first-phase API aclnnDequantSwigluQuantV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnDequantSwigluQuantV2 defaults to a deterministic implementation.

- The dimension of input x corresponding to activateDim must be a multiple of 2, and the number of dimensions of x must be greater than 1.
- When the data type of input x is INT32, weightScaleOptional cannot be null. When the data type of input x is not INT32, weightScaleOptional is not allowed and a null pointer must be passed.
- When the data type of input x is not INT32, activationScaleOptional is not allowed and a null pointer must be passed.
- When the data type of input x is not INT32, biasOptional is not allowed and a null pointer must be passed.
- When the data type of output yOut is FLOAT4_E2M1 or FLOAT4_E1M2, the last dimension of yOut must be a multiple of 2.
- When the dimension corresponding to activateDim is not the last axis of x, the last axis of output yOut must not exceed 5120.
- The sum of all elements in groupIndexOptional must not exceed the product of all axes of input x except the last axis.
- The portion of the output yOut and scaleOut that exceeds the sum of all elements in groupIndexOptional is not cleared, and the data in this memory region is garbage.
<!-- npu="A3,910b" id10 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - When groupIndexOptional is input, the memory size of input tensors supported by the operator has an upper limit, and the tail axis of x cannot exceed 7232.
  - The last dimension of x must be divisible by 64.
  - When quant_mode is dynamic, quant_offset must be empty.
<!-- end id10 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```C++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dequant_swiglu_quant_v2.h"

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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Custom construction is required based on the API.
  std::vector<int64_t> xShape = {2, 64};
  std::vector<int64_t> weightScaleShape = {1, 64};
  std::vector<int64_t> activationScaleShape = {2};
  std::vector<int64_t> biasShape = {1, 64};
  std::vector<int64_t> scaleShape = {1, 32}; // quantscale
  std::vector<int64_t> offsetShape = {1};
  std::vector<int64_t> groupIndexShape = {1};
  std::vector<int64_t> outShape = {2, 32};
  std::vector<int64_t> scaleOutShape = {2};

  void* xDeviceAddr = nullptr;
  void* weightScaleDeviceAddr = nullptr;
  void* activationScaleDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;

  void* scaleDeviceAddr = nullptr;
  void* offsetDeviceAddr = nullptr;
  void* groupIndexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* scaleOutDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* weightScale = nullptr;
  aclTensor* activationScale= nullptr;
  aclTensor* bias = nullptr;

  aclTensor* scale = nullptr;
  aclTensor* offset = nullptr;
  aclTensor* groupIndex = nullptr;
  aclTensor* out = nullptr;
  aclTensor* scaleOut = nullptr;

  std::vector<int32_t> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  std::vector<float> weightScaleData(64, 1.0);
  std::vector<float> activationScaleData(2, 1.0);
  std::vector<float> biasData(64, 1.0);
  std::vector<int64_t> groupIndexData = {1};
  std::vector<float> scaleHostData(32, 1);
  std::vector<float> offsetHostData = {1};
  std::vector<int8_t> outHostData(64, 0);
  std::vector<float> scaleOutHostData = {0, 0};

  bool activateLeft = true;
  int64_t dstType = 2;
  int64_t activateDim = -1;
  int64_t swigluMode = 1;
  float clampLimit = 7.0;
  float gluAlpha = 1.702;
  float gluBias = 1.0;

  // Create the x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT32, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the weightScale aclTensor.
  ret = CreateAclTensor(weightScaleData, weightScaleShape, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT, &weightScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the activationScale.
  ret = CreateAclTensor(activationScaleData, activationScaleShape, &activationScaleDeviceAddr, aclDataType::ACL_FLOAT, &activationScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the bias.
  ret = CreateAclTensor(biasData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the scale aclTensor.
  ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the offset aclTensor.
  ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the groupIndex aclTensor.
  ret = CreateAclTensor(groupIndexData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT64, &groupIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the scaleOut aclTensor.
  ret = CreateAclTensor(scaleOutHostData, scaleOutShape, &scaleOutDeviceAddr, aclDataType::ACL_FLOAT, &scaleOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the aclnnDequantSwigluQuantV2 first-phase API.
  ret = aclnnDequantSwigluQuantV2GetWorkspaceSize(x, weightScale, activationScale, bias, scale, nullptr, groupIndex, activateLeft, "dynamic", dstType, "rint", activateDim, swigluMode, clampLimit, gluAlpha, gluBias, out, scaleOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSwigluQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the aclnnDequantSwigluQuantV2 second-phase API.
  ret = aclnnDequantSwigluQuantV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSwigluQuantV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
  // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(scale);
  aclDestroyTensor(offset);
  aclDestroyTensor(out);
  aclDestroyTensor(scaleOut);
  // 7. Release device resources. Modify it based on the specific API definition.
  aclrtFree(xDeviceAddr);
  aclrtFree(scaleDeviceAddr);
  aclrtFree(offsetDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(scaleOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
