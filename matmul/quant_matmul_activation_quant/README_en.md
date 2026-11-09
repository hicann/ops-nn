# QuantMatmulActivationQuant

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| --- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Function: Fuses MX (Microscaling) quantized matrix multiplication, activation, and dynamic MX quantization, producing quantized output and the corresponding scales. Supports the tanh and erf forms of GELU (Gaussian Error Linear Unit), SiLU (Sigmoid Linear Unit), and SwiGLU (Swish Gated Linear Unit). For data type and transpose combinations, see [Constraints](#constraints). For an introduction to MX, see [Quantization Modes](../../docs/en/context/quant_mode_introduction.md).

- Computation flow: Perform MX quantized matrix multiplication and add bias, apply the activation, and then perform dynamic MX quantization. The formulas for each step are given below.

Batch indices are omitted below. $X_1$ and $X_2$ denote the matrices after applying the transpose settings, with shapes $(M,K)$ and $(K,N)$, respectively. M is the row count of the left matrix, K is the shared reduction dimension, and N is the column count of the right matrix.

### MX Quantized Matrix Multiplication

Split K into groups of 32 elements. Multiply each partial matrix product by the corresponding input scales and accumulate the results with bias to obtain $C$:

$$
C_{m,n}=\sum_{j=0}^{L-1}\left[
\left(\sum_{t=0}^{g_j-1}X_{1,m,jg_K+t}\,X_{2,jg_K+t,n}\right)
S_{1,m,j}\,S_{2,j,n}\right]+b_n
$$

Here, $g_M$, $g_N$, and $g_K$ are the group sizes along M, N, and K, respectively, and are currently fixed at $(1,1,32)$; $L=\lceil K/g_K\rceil$ and $g_j=\min(g_K,K-jg_K)$. $S_1$ and $S_2$ denote the scale values for group $j$ in `x1_scale` and `x2_scale`, respectively. For storage shapes, see [Shapes](#shapes). Matrix multiplication uses FP32 accumulation, and bias is added before activation. If bias is omitted, $b_n=0$.

### Activation

Apply the specified activation to $C$ to obtain $A$:

- `activation_type="gelu_tanh"`: The tanh approximation of GELU.

  $$
  A=\frac{C}{2}\left[1+\tanh\left(\sqrt{\frac{2}{\pi}}\left(C+0.044715C^3\right)\right)\right]
  $$

- `activation_type="gelu_erf"`: the erf form of GELU.

  $$
  A=\frac{C}{2}\left[1+\operatorname{erf}\left(\frac{C}{\sqrt{2}}\right)\right]
  $$

- `activation_type="silu"`: elementwise SiLU.

  $$
  A=\operatorname{SiLU}(C)=\frac{C}{1+e^{-C}}
  $$

- `activation_type="swiglu"`: Split the last dimension of $C$ into equal left and right halves, $G$ and $U$. Apply SiLU to $G$ and multiply it elementwise by $U$.

  $$
  C=[G\mid U],\qquad A=\operatorname{SiLU}(G)\odot U=\frac{G}{1+e^{-G}}\odot U
  $$

All activations above are calculated in FP32. Let $H$ be the output width after activation: $H=N$ for GELU and SiLU, and $H=N/2$ for SwiGLU.

### Dynamic MX Quantization

Convert the activation result to BFLOAT16 using Round to Nearest, ties to Even (RNE): round to the nearest representable value, resolving ties to the value whose least significant mantissa bit is 0. Then divide the last dimension into groups of 32 elements. If the last group contains fewer than 32 elements, pad it with zeros for scale calculation. The padding elements are not written to the output.

$$
V_i=\operatorname{BF16}_{RNE}(A_i),\qquad a=\max_i|V_i|,\qquad s=2^e
$$

Each group shares a scale $s$. Convert the result to the output data type $D$ according to `round_mode`:

$$
Q_i=\begin{cases}
\operatorname{FP32}(V_i/s), & D\text{ is FP8}\\
\operatorname{BF16}_{RNE}(V_i/s), & D\text{ is FP4}
\end{cases},\qquad y_i=\operatorname{Cast}_{D,\,\mathrm{round\_mode}}(Q_i)
$$

`y_scale` stores scales in FLOAT8_E8M0 format. The scale for group $j$ is written to `y_scale[..., m, j // 2, j % 2]`, and the output shape is `(..., M, ceil(H/64), 2)`.

#### Scale Calculation Algorithms

The following formulas apply to finite $a>0$. Let $p=\lfloor\log_2a\rfloor$.

- `scale_alg=0`: uses the Open Compute Project (OCP) shared-exponent algorithm. Supports FP8 and FP4.

  $$
  e=\max(-127,p-e_{\max})
  $$

  $e_{\max}$ is 8, 15, or 2 for FLOAT8_E4M3FN, FLOAT8_E5M2, or FLOAT4_E2M1, respectively.

- `scale_alg=1`: calculates the scale based on the maximum finite value of the target type. Supports only FP8.

  $$
  S=\operatorname{FP32}\left(a\cdot\operatorname{FP32}(1/F_{\max})\right),\qquad
  e=\max(-127,\lceil\log_2S\rceil)
  $$

  $F_{\max}$ is 448 for FLOAT8_E4M3FN and 57344 for FLOAT8_E5M2.

- `scale_alg=2`: calculates the scale using the range specified by `dst_type_max`. Supports only FP4. Let $d=\operatorname{FP32}(\mathrm{dst\_type\_max})$; when 0 is passed, use $d=6$.

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

## Parameters

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
| --- | --- | --- | --- | --- |
| x1 | Input | Left matrix for matrix multiplication, corresponding to X<sub>1</sub> in the formula. | FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1 | ND |
| x2 | Input | Right matrix for matrix multiplication, corresponding to X<sub>2</sub> in the formula. NZ does not support FLOAT8_E5M2. | FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1 | ND, FRACTAL_NZ |
| bias | Optional input | <ul><li>Bias added before activation, corresponding to b in the formula.</li><li>No bias is added if omitted.</li></ul> | FLOAT32 | ND |
| x1_scale | Optional input | <ul><li>Group scales for x1, corresponding to S<sub>1</sub> in the formula.</li><li>This operator supports only MX mode. The corresponding group scales must be provided at execution.</li></ul> | FLOAT8_E8M0 | ND |
| x2_scale | Optional input | <ul><li>Group scales for x2, corresponding to S<sub>2</sub> in the formula.</li><li>This operator supports only MX mode. The corresponding group scales must be provided at execution.</li></ul> | FLOAT8_E8M0 | ND |
| y | Output | Dynamic quantization result, corresponding to y in the formula. Its data type must match that of x1. | FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1 | ND |
| y_scale | Output | Scale for each group of 32 output elements, corresponding to s in the formula. | FLOAT8_E8M0 | ND |
| transpose_x1 | Optional attribute | <ul><li>Whether to transpose the last two dimensions of x1: false means no transpose, and true means transpose.</li><li>Defaults to false.</li></ul> | BOOL | - |
| transpose_x2 | Optional attribute | <ul><li>Whether to transpose the last two dimensions of x2: false means no transpose, and true means transpose.</li><li>Defaults to false.</li></ul> | BOOL | - |
| group_size | Optional attribute | <ul><li>Encoded group sizes along M, N, and K for input quantization. Supports only 0 or 4295032864, both representing [1,1,32].</li><li>Defaults to 0.</li></ul> | INT64 | - |
| activation_type | Optional attribute | <ul><li>Activation type. Supports "gelu_tanh", "gelu_erf", "silu", and "swiglu".</li><li>Defaults to "gelu_tanh".</li></ul> | STRING | - |
| y_dtype | Optional attribute | <ul><li>Data type enumeration for y. Supports the enumeration values for FLOAT8_E4M3FN, FLOAT8_E5M2, and FLOAT4_E2M1. The specified data type must match that of x1.</li><li>Defaults to the enumeration value for FLOAT8_E4M3FN.</li></ul> | INT64 | - |
| quant_mode | Optional attribute | <ul><li>Quantization mode. Supports only "mx".</li><li>Defaults to "mx".</li></ul> | STRING | - |
| round_mode | Optional attribute | <ul><li>Rounding mode for conversion to the output data type. FP8 supports only "rint", which rounds to the nearest representable value and resolves ties to the value whose least significant mantissa bit is 0.</li><li>FP4 also supports "floor" (round toward negative infinity) and "round" (round to the nearest representable value, resolving ties away from zero).</li><li>Defaults to "rint".</li></ul> | STRING | - |
| scale_alg | Optional attribute | <ul><li>Algorithm for calculating output scales. FP8 supports 0 and 1; FP4 supports 0 and 2.</li><li>For algorithm definitions, see <a href="#scale-calculation-algorithms">Scale Calculation Algorithms</a>. Defaults to 0.</li></ul> | INT64 | - |
| dst_type_max | Optional attribute | <ul><li>Reference upper bound for calculating output scales when scale_alg=2, corresponding to d in the formula.</li><li>Supports 0.0 (equivalent to 6.0) or a finite value in [6.0,12.0].</li><li>Defaults to 0.0. Ignored by other algorithms.</li></ul> | FLOAT32 | - |

## Constraints

The following input requirements apply when the operator executes. For interface layout restrictions, see [aclnnQuantMatmulActivationQuant](docs/aclnnQuantMatmulActivationQuant_en.md) and [aclnnQuantMatmulActivationQuantWeightNz](docs/aclnnQuantMatmulActivationQuantWeightNz_en.md).

- x1 and x2 must each have 2 to 6 logical dimensions. Empty tensors are not supported, and every dimension must be a positive integer.
- Each logical dimension of x1/x2 and the product of the broadcast batch dimensions must not exceed $2^{32}-1$. The number of output elements before activation must not exceed $2^{63}-1$.

The matrices and output support the following combinations:

| x1 | x2 | y | x2 format |
| --- | --- | --- | --- |
| FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E4M3FN | ND, FRACTAL_NZ |
| FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT8_E5M2 | ND, FRACTAL_NZ |
| FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT8_E4M3FN | ND |
| FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT8_E5M2 | ND |
| FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT4_E2M1 | ND, FRACTAL_NZ |

### Activations and Transposes

<a id="activation-capability-comparison-table"></a>

In the table below, FP8 refers to FLOAT8_E4M3FN or FLOAT8_E5M2, and FP4 refers to FLOAT4_E2M1. The combinations must also satisfy the data type table above.

| activation_type | Data type | Output width H | Requirement on N | transpose_x1 when x2 is ND | transpose_x1 when x2 is NZ |
| --- | --- | --- | --- | --- | --- |
| "gelu_tanh", "gelu_erf" | FP8 | N | $N>0$ | false, true | false, true |
| "gelu_tanh", "gelu_erf" | FP4 | N | Positive even integer | false, true | false |
| "silu" | FP8 | N | $N>0$ | false, true | false |
| "swiglu" | FP8 | N/2 | Positive multiple of 64 | false, true | false |

Both false and true are supported for `transpose_x2`. When x2 uses NZ, K and N must both be greater than 1.

FP4 supports only GELU and requires $K>2$ and an even N. For FP4 inputs x1 and x2, the innermost dimension in a contiguous storage layout must be even: check K for non-transposed x1 and M for transposed x1; check N for non-transposed x2 and K for transposed x2.

### Shapes

`batch1` and `batch2` denote the zero to four leading batch dimensions of the inputs. `batch` denotes the result of right-aligned broadcasting. Omit these parts when no batch dimensions are present. Let `K64=ceil(K/64)`, where ceil rounds up to the nearest integer. All shapes below are logical shapes; FP4 dimensions count 4-bit elements.

| Tensor | Not Transposed | Transposed |
| --- | --- | --- |
| x1 | `(batch1, M, K)` | `(batch1, K, M)` |
| x1_scale | `(batch1, M, K64, 2)` | `(batch1, K64, M, 2)` |
| x2 | `(batch2, K, N)` | `(batch2, N, K)` |
| x2_scale | `(batch2, K64, N, 2)` | `(batch2, N, K64, 2)` |

- The two K dimensions must match. Each input scale's batch dimensions must exactly match those of its matrix.
- Each group of 32 K elements has one input scale. Scales for two consecutive groups are stored in order in the trailing dimension of size 2.
- The shape of `y` is `(batch, M, H)`, and the shape of `y_scale` is `(batch, M, ceil(H/64), 2)`. H is determined by the activation type.
- The shape of bias is `(N,)`. Only for a three-dimensional output, `(B, 1, N)` is also supported, where B must equal the broadcast batch size. For SwiGLU, the bias length remains N, the width before activation.

Nonzero `group_size` values are encoded as `(groupSizeM << 32) | (groupSizeN << 16) | groupSizeK`, with the highest 16 bits set to 0. This operator uses fixed group sizes of `[1,1,32]`.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn API | [GELU example](examples/arch35/test_aclnn_quant_matmul_gelu_mx_quant.cpp), [SwiGLU example](examples/arch35/test_aclnn_quant_matmul_swiglu_mx_quant.cpp) | Calls [aclnnQuantMatmulActivationQuant](docs/aclnnQuantMatmulActivationQuant_en.md) with x2 in ND format. |
| aclnn API | [GELU example with NZ weights](examples/arch35/test_aclnn_quant_matmul_gelu_mx_quant_weight_nz.cpp) | Calls [aclnnQuantMatmulActivationQuantWeightNz](docs/aclnnQuantMatmulActivationQuantWeightNz_en.md) with x2 in FRACTAL_NZ format. |
| GE graph mode | - | Builds and invokes a graph using the [QuantMatmulActivationQuant operator IR](op_graph/quant_matmul_activation_quant_proto.h). |
| PyTorch API | [FP8 and FP4 examples](docs/torchapi_quant_matmul_activation_quant_en.md#example) | Calls [cann_ops_nn.quant_matmul_activation_quant](docs/torchapi_quant_matmul_activation_quant_en.md). For extension installation, see the [PyTorch extension instructions](../../torch_extension/README_en.md). |

## References

- For test execution instructions, see the [test guide](tests/README_en.md).
