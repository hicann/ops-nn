# aclnnQuantMatmulActivationQuant

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: This API fuses MX (Microscaling) quantized matrix multiplication, activation, and dynamic MX quantization to return the quantized result and its scales. Supported activations are GELU (Gaussian Error Linear Unit, tanh/erf forms), SiLU (Sigmoid Linear Unit), and SwiGLU (Swish Gated Linear Unit). Both x1 and x2 use ND. See [Constraints](#constraints) for data types and transpose combinations.

<!-- npu="950" id7 -->

- Computation: Perform MX matrix multiplication and add the bias, apply the activation, and then dynamically quantize the result in MX mode.

Batch indices are omitted below. $X_1$ and $X_2$ denote the matrices after applying the transpose settings, with shapes $(M,K)$ and $(K,N)$, respectively. M is the row count of the left matrix, K is the shared reduction dimension, and N is the column count of the right matrix.

### MX Quantized Matrix Multiplication

Split K into groups of 32 elements. Multiply each partial matrix product by the corresponding input scales and accumulate the results with bias to obtain $C$:

$$
C_{m,n}=\sum_{j=0}^{L-1}\left[
\left(\sum_{t=0}^{g_j-1}X_{1,m,jg_K+t}\,X_{2,jg_K+t,n}\right)
S_{1,m,j}\,S_{2,j,n}\right]+b_n
$$

Here, $g_M$, $g_N$, and $g_K$ are the group sizes along M, N, and K, respectively, and are currently fixed at $(1,1,32)$; $L=\lceil K/g_K\rceil$ and $g_j=\min(g_K,K-jg_K)$. $S_1$ and $S_2$ denote the scale values for group $j$ in `x1ScaleOptional` and `x2Scale`, respectively. For storage shapes, see [Shapes](#shapes). Matrix multiplication uses FP32 accumulation, and bias is added before activation. If bias is omitted, $b_n=0$.

### Activation

Apply the specified activation to $C$ to obtain $A$:

- `activationType="gelu_tanh"`: The tanh approximation of GELU.

  $$
  A=\frac{C}{2}\left[1+\tanh\left(\sqrt{\frac{2}{\pi}}\left(C+0.044715C^3\right)\right)\right]
  $$

- `activationType="gelu_erf"`: The erf form of GELU.

  $$
  A=\frac{C}{2}\left[1+\operatorname{erf}\left(\frac{C}{\sqrt{2}}\right)\right]
  $$

- `activationType="silu"`: Elementwise SiLU.

  $$
  A=\operatorname{SiLU}(C)=\frac{C}{1+e^{-C}}
  $$

- `activationType="swiglu"`: Split the last dimension of $C$ into equal left and right halves, $G$ and $U$. Apply SiLU to $G$ and multiply it elementwise by $U$.

  $$
  C=[G\mid U],\qquad A=\operatorname{SiLU}(G)\odot U=\frac{G}{1+e^{-G}}\odot U
  $$

All activations above are calculated in FP32. Let $H$ be the output width after activation: $H=N$ for GELU and SiLU, and $H=N/2$ for SwiGLU.

### Dynamic MX Quantization

Convert the activation result to BF16 using Round to Nearest, ties to Even (RNE): round to the nearest representable value, resolving ties to the value whose least significant mantissa bit is 0. Then divide the last dimension into groups of 32 elements. If the last group contains fewer than 32 elements, pad it with zeros for scale calculation. The padding elements are not written to the output.

$$
V_i=\operatorname{BF16}_{RNE}(A_i),\qquad a=\max_i|V_i|,\qquad s=2^e
$$

Each group shares a scale $s$. Convert the normalized values to output type $D$ using `roundMode`:

$$
Q_i=\begin{cases}
\operatorname{FP32}(V_i/s), & D\text{ is FP8}\\
\operatorname{BF16}_{RNE}(V_i/s), & D\text{ is FP4}
\end{cases},\qquad (yOut)_i=\operatorname{Cast}_{D,\,\mathrm{roundMode}}(Q_i)
$$

`yScaleOut` stores scales as FLOAT8_E8M0. The scale for group $j$ is written to `yScaleOut[..., m, j // 2, j % 2]`; its shape is `(..., M, ceil(H/64), 2)`.

#### Scale Calculation Algorithms

The following formulas apply to finite $a>0$. Let $p=\lfloor\log_2a\rfloor$.

- `scaleAlg=0`: The OCP (Open Compute Project) shared-exponent algorithm, available for FP8 and FP4.

  $$
  e=\max(-127,p-e_{\max})
  $$

  $e_{\max}$ is 8, 15, or 2 for FLOAT8_E4M3FN, FLOAT8_E5M2, or FLOAT4_E2M1, respectively.

- `scaleAlg=1`: Calculate the scale from the largest finite value of the target type. Available for FP8 only.

  $$
  S=\operatorname{FP32}\left(a\cdot\operatorname{FP32}(1/F_{\max})\right),\qquad
  e=\max(-127,\lceil\log_2S\rceil)
  $$

  $F_{\max}$ is 448 for FLOAT8_E4M3FN and 57344 for FLOAT8_E5M2.

- `scaleAlg=2`: Calculate the scale from the range specified by `dstTypeMax`. Available for FP4 only. Let $d=\operatorname{FP32}(\mathrm{dstTypeMax})$; an input of 0 selects $d=6$.

  For $d=6$ or $d=7$:

  $$
  e=\max\left(-127,\begin{cases}
  p-1, & a>d\cdot2^{p-2}\\
  p-2, & a\le d\cdot2^{p-2}
  \end{cases}\right)
  $$

  For other values of $d$ in $[6,12]$:

  $$
  S=\operatorname{FP32}\left(a\cdot\operatorname{FP32}(1/d)\right),\qquad
  e=\max(-127,\lceil\log_2S\rceil)
  $$

E8M0 encodes $2^e$ as the byte $e+127$. Special cases are handled as follows:

- An all-zero group produces zeros and a scale encoding of 0. Padding slots in the output scales that have no corresponding data group are also set to 0. This encoding represents $2^{-127}$.
- If $S=0$, use $e=-127$. Algorithm 0, and algorithm 2 with $d=6$ or $d=7$, set the group's quantized results to 0 when the scale encoding is 0.
- For a group containing NaN or Inf, the scale encoding is 255 (NaN).

<!-- end id7 -->

## Function Prototype

Each operator provides a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call `aclnnQuantMatmulActivationQuantGetWorkspaceSize` to obtain the workspace size required for computation and the executor containing the operator's computation flow, then call `aclnnQuantMatmulActivationQuant` to execute the computation.

```cpp
aclnnStatus aclnnQuantMatmulActivationQuantGetWorkspaceSize(
  const aclTensor* x1,
  const aclTensor* x2,
  const aclTensor* x1ScaleOptional,
  const aclTensor* x2Scale,
  const aclTensor* biasOptional,
  bool             transposeX1,
  bool             transposeX2,
  int64_t          groupSize,
  char*            activationType,
  char*            quantMode,
  char*            roundMode,
  int64_t          scaleAlg,
  double           dstTypeMax,
  aclTensor*       yOut,
  aclTensor*       yScaleOut,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnQuantMatmulActivationQuant(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnQuantMatmulActivationQuantGetWorkspaceSize

- **Parameters**

  All tensor data resides on the device. Only `biasOptional` may be nullptr; all other tensor parameters are required and must be nonempty.

  <table style="table-layout: fixed; width: 1500px">
    <colgroup>
      <col style="width: 210px">
      <col style="width: 100px">
      <col style="width: 240px">
      <col style="width: 390px">
      <col style="width: 230px">
      <col style="width: 120px">
      <col style="width: 110px">
      <col style="width: 100px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th>Format</th>
        <th>Dimensions (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x1 (const aclTensor*)</td>
        <td>Input</td>
        <td>Left matrix, corresponding to X<sub>1</sub> in the formulas.</td>
        <td><ul> <li>Required and nonempty.</li> <li>See <a href="#shapes">Shapes</a> and <a href="#data-types-and-activations">Data Types and Activations</a> for shape and transpose restrictions.</li> </ul></td>
        <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
        <td>ND</td>
        <td>2–6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2 (const aclTensor*)</td>
        <td>Input</td>
        <td>Right matrix, corresponding to X<sub>2</sub> in the formulas.</td>
        <td><ul> <li>Required and nonempty.</li> <li>See <a href="#shapes">Shapes</a> for logical shapes and <a href="#layout-and-storage">Layout and Storage</a> for storage requirements.</li> </ul></td>
        <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
        <td>ND</td>
        <td>2–6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x1ScaleOptional (const aclTensor*)</td>
        <td>Input</td>
        <td>Scales for x1 groups along K, corresponding to S<sub>1</sub>.</td>
        <td><ul> <li>This API supports only MX mode; this input must be supplied and must not be nullptr.</li> <li>Must be nonempty; batch dimensions must match x1.</li> </ul></td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>3–7</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2Scale (const aclTensor*)</td>
        <td>Input</td>
        <td>Scales for x2 groups along K, corresponding to S<sub>2</sub>.</td>
        <td><ul> <li>Required and nonempty.</li> <li>Batch dimensions must match x2.</li> </ul></td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>3–7</td>
        <td>√</td>
      </tr>
      <tr>
        <td>biasOptional (const aclTensor*)</td>
        <td>Optional input</td>
        <td>Bias added before activation, corresponding to b.</td>
        <td><ul> <li>Shape: (N,); (B,1,N) is also supported for a 3D output.</li> <li>Pass nullptr to omit the bias. A supplied tensor must be nonempty.</li> </ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1 or 3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>transposeX1 (bool)</td>
        <td>Input</td>
        <td>Whether to transpose the last two dimensions of x1.</td>
        <td><ul> <li>false means no transpose, and true means transpose.</li> <li>For supported data type and activation combinations, see <a href="#data-types-and-activations">Data Types and Activations</a>.</li> <li>For supported transpose combinations, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX2 (bool)</td>
        <td>Input</td>
        <td>Whether to transpose the last two dimensions of x2.</td>
        <td><ul> <li>false means no transpose, and true means transpose.</li> <li>For supported combinations, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize (int64_t)</td>
        <td>Input</td>
        <td>Encoded input quantization group sizes along M, N, and K.</td>
        <td><ul> <li>Only 0 and 4295032864 are supported; both specify [1,1,32].</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>activationType (char*)</td>
        <td>Input</td>
        <td>Activation type.</td>
        <td><ul> <li>Supports "gelu_tanh", "gelu_erf", "silu", and "swiglu".</li> <li>Must not be nullptr.</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMode (char*)</td>
        <td>Input</td>
        <td>Quantization mode.</td>
        <td><ul> <li>Only "mx" is supported. Must not be nullptr.</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>roundMode (char*)</td>
        <td>Input</td>
        <td>Rounding mode for output quantization.</td>
        <td><ul> <li>FP8 supports only "rint": round to the nearest representable value, with ties to even.</li> <li>FP4 also supports "floor" (round toward negative infinity) and "round" (round to the nearest representable value, with ties away from zero).</li> <li>Must not be nullptr.</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleAlg (int64_t)</td>
        <td>Input</td>
        <td>Algorithm for calculating output scales.</td>
        <td><ul> <li>FP8 supports 0 and 1; FP4 supports 0 and 2.</li> <li>0 uses the OCP shared exponent; 1 uses the FP8 maximum finite value; 2 uses the range specified by dstTypeMax.</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dstTypeMax (double)</td>
        <td>Input</td>
        <td>Reference maximum used to calculate output scales when scaleAlg=2, corresponding to d in the formulas.</td>
        <td><ul> <li>Applies only when scaleAlg=2: 0.0 selects 6.0; finite values in [6.0,12.0] are also supported.</li> <li>Ignored by the other algorithms.</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOut (aclTensor*)</td>
        <td>Output</td>
        <td>Dynamically quantized result, corresponding to yOut in the formulas.</td>
        <td><ul> <li>Required and nonempty.</li> <li>Must have the same data type as x1. Shape: (batch,M,H).</li> </ul></td>
        <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
        <td>ND (see <a href="#layout-and-storage">Layout and Storage</a> for compatible formats)</td>
        <td>2–6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>yScaleOut (aclTensor*)</td>
        <td>Output</td>
        <td>Scales for output groups, corresponding to s in the formulas.</td>
        <td><ul> <li>Required and nonempty.</li> <li>Shape: (batch,M,ceil(H/64),2).</li> </ul></td>
        <td>FLOAT8_E8M0</td>
        <td>ND (see <a href="#layout-and-storage">Layout and Storage</a> for compatible formats)</td>
        <td>3–7</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to allocate on the device, in bytes.</td>
        <td><ul> <li>Must not be nullptr.</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor containing the operator's computation flow.</td>
        <td><ul> <li>Must not be nullptr.</li> </ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

  A nonzero `groupSize` is encoded as `(groupSizeM << 32) | (groupSizeN << 16) | groupSizeK`; the upper 16 bits must be zero. Use int64_t values for the shifts.

- **Return Value**

  aclnnStatus: Returns a status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API validates the input parameters and reports errors in the following cases:

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 300px">
      <col style="width: 150px">
      <col style="width: 550px">
    </colgroup>
    <thead>
      <tr>
        <th>Return/Status Code</th>
        <th>Value</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>x1, x2, x1ScaleOptional, x2Scale, yOut, yScaleOut, workspaceSize, or executor is a null pointer.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>A tensor data type, format, shape, broadcast relationship, stride, or accessed storage range violates the constraints; a string parameter is null; or an attribute value or combination is invalid.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>Failed to create the operator executor.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>An internal tensor operation returned a null pointer during tensor creation, preprocessing, operator computation, or result copying.</td>
      </tr>
    </tbody>
  </table>

## aclnnQuantMatmulActivationQuant

- **Parameters**

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 220px">
      <col style="width: 100px">
      <col style="width: 680px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>workspace (void*)</td>
        <td>Input</td>
        <td>Address of the workspace memory allocated on the device.</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t)</td>
        <td>Input</td>
        <td>Size of the workspace allocated on the device, in bytes, obtained from the first-phase API aclnnQuantMatmulActivationQuantGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor*)</td>
        <td>Input</td>
        <td>Op executor containing the operator's computation flow.</td>
      </tr>
      <tr>
        <td>stream (aclrtStream)</td>
        <td>Input</td>
        <td>Stream specified for executing the task.</td>
      </tr>
    </tbody>
  </table>

- **Return Value**

  aclnnStatus: Returns a status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  ACLNN_ERR_INNER (561000) is returned if executor execution fails.

## Constraints

- x1 and x2 have 2 to 6 logical dimensions, all of which must be known positive integers.
- Each logical dimension of x1/x2 and the product of the broadcast batch dimensions must not exceed $2^{32}-1$. The number of output elements before activation must not exceed $2^{63}-1$.

### Data Types and Activations

Input and output scales use FLOAT8_E8M0. biasOptional must use FLOAT32 when supplied; pass nullptr to omit the bias. Supported matrix and output combinations are:

| x1 | x2 | yOut |
| --- | --- | --- |
| FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E4M3FN |
| FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT8_E5M2 |
| FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT8_E4M3FN |
| FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT8_E5M2 |
| FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT4_E2M1 |

In the table below, FP8 refers to FLOAT8_E4M3FN or FLOAT8_E5M2, and FP4 refers to FLOAT4_E2M1. The combinations must also satisfy the data type table above.

| activationType | Data Type | Output Width H | N Requirement | transposeX1 |
| --- | --- | --- | --- | --- |
| "gelu_tanh", "gelu_erf" | FP8 | N | N > 0 | false, true |
| "gelu_tanh", "gelu_erf" | FP4 | N | Positive even integer | false, true |
| "silu" | FP8 | N | N > 0 | false, true |
| "swiglu" | FP8 | N/2 | Positive multiple of 64 | false, true |

`transposeX2` supports both false and true. FP4 supports GELU only and requires K > 2 and even N. For FP4 inputs, the innermost axis of the contiguous storage layout must be even: K for non-transposed x1, M for transposed x1, N for non-transposed x2, and K for transposed x2.

### Shapes

`batch1` and `batch2` denote the zero to four leading batch dimensions of the inputs. `batch` denotes the result of right-aligned broadcasting. Omit these parts when no batch dimensions are present. Let `K64=ceil(K/64)`, where ceil rounds up to the nearest integer. All shapes below are logical shapes; FP4 dimensions count 4-bit elements.

| Tensor | Not Transposed | Transposed |
| --- | --- | --- |
| x1 | `(batch1, M, K)` | `(batch1, K, M)` |
| x1ScaleOptional | `(batch1, M, K64, 2)` | `(batch1, K64, M, 2)` |
| x2 | `(batch2, K, N)` | `(batch2, N, K)` |
| x2Scale | `(batch2, K64, N, 2)` | `(batch2, N, K64, 2)` |

- The two K dimensions must match. Each input scale's batch dimensions must exactly match those of its matrix.
- Each group of 32 K elements has one input scale. Scales for two consecutive groups are stored in order in the trailing dimension of size 2.
- `yOut` has shape `(batch, M, H)` and `yScaleOut` has shape `(batch, M, ceil(H/64), 2)`. The activation determines H.
- `biasOptional` has shape `(N,)`. Only for a 3D output, `(B, 1, N)` is also supported, where B equals the broadcast batch size. For SwiGLU, the biasOptional length remains the pre-activation width N.

### Layout and Storage

- yOut and yScaleOut support ND and the ND-compatible base formats NCHW, NHWC, HWCN, NCDHW, NDHWC, and NCL. Private formats are not supported. Output shapes must still follow the rules above.
- ND input and output strides and offsets must be nonnegative. Every access determined by the shape, strides, and offset must be within the underlying storage. Slices with a nonzero offset are supported when these requirements hold. FP4 offsets count logical elements and must be even.
- x1, x2, both input scales, and biasOptional support [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md). Outputs also support non-contiguous tensors.
- When a matrix is passed as a view transposing its last two axes, its scale must swap the third-last and second-last axes, leaving the trailing dimension of size 2 unchanged. A transpose view and a true transposeX1/transposeX2 flag apply two transpositions, which cancel each other. The transpose restrictions above apply to the effective transpose state.

### Determinism and Batch Consistency

<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>:
  - Deterministic computation: aclnnQuantMatmulActivationQuant uses a deterministic implementation by default.
  - Batch consistency: Uses a [batch-consistent implementation](../../../docs/en/context/batch_consistency.md) by default under the following conditions. In the same runtime environment, keep N, K, data types, layouts, and activation/quantization attributes unchanged. For a given token, keep its x1 and x1ScaleOptional data and the corresponding x2, x2Scale, and biasOptional data unchanged. Changing only the batch size, M, or the positions or combinations of tokens preserves bitwise-identical yOut and yScaleOut values for that token. No additional switch is required.
<!-- end id8 -->

## Example

This example uses FLOAT8_E4M3FN inputs, GELU (tanh), and `scaleAlg=0`.

The following sample code is provided for reference. For details on compilation and execution, see [Compiling and Running a Sample](../../../docs/en/context/compile_and_run_sample.md).

See also the [GELU sample](../examples/arch35/test_aclnn_quant_matmul_gelu_mx_quant.cpp) and [SwiGLU sample](../examples/arch35/test_aclnn_quant_matmul_swiglu_mx_quant.cpp).

```cpp
// ND weights: QuantMatmul + GELU(tanh) + MXFP8 quantization.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_matmul_activation_quant.h"
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
    // Initialize the runtime resources.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);
    return 0;
}
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Copy host data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Calculate strides for a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // Create the aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return 1);
    return 0;
}
void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int AclnnQuantMatmulGeluMxQuantTest(aclrtStream stream)
{
    int ret = ACL_SUCCESS;
    // 2. Construct the input and output tensors.
    int64_t m = 5;
    int64_t k = 64;
    int64_t n = 128;
    bool transposeX1 = false;
    bool transposeX2 = false;
    constexpr int64_t MX_SCALE_BLOCK_SIZE = 64; // Alignment used to calculate scale shapes.
    std::vector<int64_t> x1Shape = {m, k};
    std::vector<int64_t> x2Shape = {k, n};
    std::vector<int64_t> x1ScaleShape = {m, (k + MX_SCALE_BLOCK_SIZE - 1) / MX_SCALE_BLOCK_SIZE, 2};
    std::vector<int64_t> x2ScaleShape = {(k + MX_SCALE_BLOCK_SIZE - 1) / MX_SCALE_BLOCK_SIZE, n, 2};
    std::vector<int64_t> outShape = {m, n};
    std::vector<int64_t> outScaleShape = {m, (n + MX_SCALE_BLOCK_SIZE - 1) / MX_SCALE_BLOCK_SIZE, 2};
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* x1ScaleDeviceAddr = nullptr;
    void* x2ScaleDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* outScaleDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* x1Scale = nullptr;
    aclTensor* x2Scale = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* out = nullptr;
    aclTensor* outScale = nullptr;
    std::vector<uint8_t> x1HostData(m * k, 0b00111000); // 1.0 encoded as FLOAT8_E4M3FN.
    std::vector<uint8_t> x2HostData(k * n, 0b00111000); // 1.0 encoded as FLOAT8_E4M3FN.
    std::vector<uint8_t> x1ScaleHostData(GetShapeSize(x1ScaleShape),
                                         0b01111111); // 1.0 encoded as FLOAT8_E8M0.
    std::vector<uint8_t> x2ScaleHostData(GetShapeSize(x2ScaleShape),
                                         0b01111111); // 1.0 encoded as FLOAT8_E8M0.
    std::vector<uint8_t> outHostData(GetShapeSize(outShape), 0);
    std::vector<uint8_t> outScaleHostData(GetShapeSize(outScaleShape), 0);
    // Create x1.
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create x2.
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create x1Scale.
    ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x1Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create x2Scale.
    ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x2Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the quantized output.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the output scales.
    ret = CreateAclTensor(outScaleHostData, outScaleShape, &outScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                          &outScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outScaleTensorPtr(outScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outScaleDeviceAddrPtr(outScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator APIs.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int64_t groupSizeValue = 4295032864;
    char activationType[] = "gelu_tanh";
    char quantMode[] = "mx";
    char roundMode[] = "rint";
    ret = aclnnQuantMatmulActivationQuantGetWorkspaceSize(x1, x2, x1Scale, x2Scale, bias, transposeX1, transposeX2,
                                                          groupSizeValue, activationType, quantMode, roundMode, 0, 0.0,
                                                          out, outScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulActivationQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate the workspace requested by the first phase.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second phase of aclnnQuantMatmulActivationQuant.
    ret = aclnnQuantMatmulActivationQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulActivationQuant failed. ERROR: %d\n", ret); return ret);
    // 4. Wait for stream execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Copy the output data from device to host.
    auto size = GetShapeSize(outShape);
    std::vector<uint8_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%lld] (FP8 raw byte): %u\n", static_cast<long long>(i), static_cast<unsigned>(resultData[i]));
    }
    size = GetShapeSize(outScaleShape);
    std::vector<uint8_t> scaleData(size, 0);
    ret = aclrtMemcpy(scaleData.data(), scaleData.size() * sizeof(scaleData[0]), outScaleDeviceAddr,
                      size * sizeof(scaleData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale result from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("scale[%lld] (E8M0 raw byte): %u\n", static_cast<long long>(i), static_cast<unsigned>(scaleData[i]));
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. Initialize the device and stream.
    // Select the device ID for the target environment.
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = AclnnQuantMatmulGeluMxQuantTest(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("AclnnQuantMatmulGeluMxQuantTest failed. ERROR: %d\n", ret);
    }
    Finalize(deviceId, stream);
    return ret;
}
```
