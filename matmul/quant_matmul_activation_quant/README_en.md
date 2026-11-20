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

- Function: Fuses MX (Microscaling) quantized matrix multiplication, activation, and dynamic MX quantization, producing quantized output and the corresponding scales. Supports the tanh and erf forms of GELU (Gaussian Error Linear Unit), SiLU (Sigmoid Linear Unit), SwiGLU (Swish Gated Linear Unit), and sigmoid followed by elementwise multiplication with `x3`. For data type and transpose combinations, see [Constraints](#constraints). For an introduction to MX, see [Quantization Modes](../../docs/en/context/quant_mode_introduction.md).

- Both V2 aclnn interfaces (ND and FRACTAL_NZ weights) add `x3Optional` for elementwise multiplication with the sigmoid result. V1 and V2 support GELU, SiLU, and SwiGLU. See [Invocation Description](#invocation-description) for the four interfaces.

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

- `activation_type="sigmoid"`: Apply sigmoid to $C$ and multiply elementwise by the BFLOAT16 tensor $B$, provided by `x3`.

  $$
  A=\operatorname{sigmoid}(C)\odot\operatorname{FP32}(B)
  $$

All activations above are calculated in FP32. Let $H$ be the output width after activation: $H=N$ for GELU, SiLU, and sigmoid, and $H=N/2$ for SwiGLU.

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
| x1 | Input | Left matrix in the product, denoted by $X_1$. | FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1 | ND |
| x2 | Input | Right matrix in the product, denoted by $X_2$. FRACTAL_NZ supports FLOAT8_E4M3FN and FLOAT4_E2M1. | FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1 | ND, FRACTAL_NZ |
| bias | Optional input | Bias $b$ added before activation. Omitting it gives zero bias. | FLOAT32 | ND |
| x1_scale | Optional input | Group scales $S_1$ along K for `x1`. Required for MX, the only supported quantization mode. | FLOAT8_E8M0 | ND |
| x2_scale | Optional input | Group scales $S_2$ along K for `x2`. Required for MX, the only supported quantization mode. | FLOAT8_E8M0 | ND |
| x3 | Optional input | Tensor $B$ multiplied elementwise with the sigmoid result. Supply it for `activation_type="sigmoid"` and omit it for other activations. See [Sigmoid Scenario](#sigmoid-scenario). | BFLOAT16 | ND |
| y | Output | Dynamically quantized result $y$. Its data type must match `x1`. | FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1 | ND |
| y_scale | Output | Scale $s$ shared by each group of 32 output elements. | FLOAT8_E8M0 | ND |
| transpose_x1 | Optional attribute | Whether to transpose the final two dimensions of `x1`. Use `false` (default) to keep their order or `true` to swap them. | BOOL | - |
| transpose_x2 | Optional attribute | Whether to transpose the final two dimensions of `x2`. Use `false` (default) to keep their order or `true` to swap them. | BOOL | - |
| group_size | Optional attribute | Encoded group sizes along M, N, and K for input quantization. Both `0` (default) and `4295032864` select `[1, 1, 32]`. | INT64 | - |
| activation_type | Optional attribute | Activation function. Supports `"gelu_tanh"` (default), `"gelu_erf"`, `"silu"`, `"swiglu"`, and `"sigmoid"`. | STRING | - |
| y_dtype | Optional attribute | Data type identifier for `y`. It must match the data type of `x1`. Supported values are `35` (FLOAT8_E5M2), `36` (FLOAT8_E4M3FN, default), and `40` (FLOAT4_E2M1). | INT64 | - |
| quant_mode | Optional attribute | Quantization mode. The only supported value is `"mx"` (default). | STRING | - |
| round_mode | Optional attribute | Rounding mode for output quantization. FP8 supports `"rint"` (default, round to nearest with ties to even). FP4 also supports `"floor"` (round toward negative infinity) and `"round"` (round to nearest with ties away from zero). | STRING | - |
| scale_alg | Optional attribute | Algorithm for computing output scales. FP8 supports `0` and `1`. FP4 supports `0` and `2`. The default is `0`. See [Scale Calculation Algorithms](#scale-calculation-algorithms). | INT64 | - |
| dst_type_max | Optional attribute | Reference upper bound $d$ for computing the scale with `scale_alg=2`. Accepts `0.0` (default, treated as `6.0`) or a finite value in $[6,12]$. Other algorithms ignore this attribute. | FLOAT32 | - |

## Constraints

The following input requirements apply when the operator executes. For interface layout restrictions, see [aclnnQuantMatmulActivationQuantV2](docs/aclnnQuantMatmulActivationQuantV2_en.md) and [aclnnQuantMatmulActivationQuantWeightNzV2](docs/aclnnQuantMatmulActivationQuantWeightNzV2_en.md).

- x1 and x2 must each have 2 to 6 logical dimensions. Empty tensors are not supported, and every dimension must be a positive integer.
- Each logical dimension of every input and output tensor, including scales and any supplied optional tensors, must be in the range $[1,2^{32}-1]$.
- Storage dimensions must be positive. For each tensor, multiply the dimensions of its logical shape and its storage shape separately. Each element count must be at most $2^{63}-1$. The NZ storage count includes alignment padding. Count FP4 data in 4-bit elements.
- The product of the broadcast batch dimensions, `batchSize`, must not exceed $2^{32}-1$. This product excludes M, K, and N; `batchSize` is 1 when there are no batch dimensions.
- The number of output elements before activation, `batchSize * M * N`, must not exceed $2^{63}-1$, where N is the number of columns in the matrix multiplication output before activation.

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
| "gelu_tanh", "gelu_erf" | FP8 | N | ND $N>0$, NZ $N>1$ | false, true | false, true |
| "gelu_tanh", "gelu_erf" | FP4 | N | Positive even integer | false, true | false |
| "silu" | FP8 | N | ND $N>0$, NZ $N>1$ | false, true | false |
| "sigmoid" | FP8 | N | ND $N>0$, NZ $N>1$ | false, true | false |
| "swiglu" | FP8 | N/2 | Positive multiple of 64 | false, true | false |

All combinations support both `false` and `true` for `transpose_x2`. NZ weights also require $K>1$.

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
- The shape of `y` is `(batch, M, H)`, and the shape of `y_scale` is `(batch, M, ceil(H/64), 2)`. H is determined by the activation type. Both outputs must be contiguous.
- The shape of bias is `(N,)`. Only for a three-dimensional output, `(B, 1, N)` is also supported, where B must equal the broadcast batch size. For SwiGLU, the bias length remains N, the width before activation.

Nonzero `group_size` values are encoded as `(groupSizeM << 32) | (groupSizeN << 16) | groupSizeK`, with the highest 16 bits set to 0. This operator uses fixed group sizes of `[1,1,32]`.

### Sigmoid Scenario

For `activation_type="sigmoid"`:

- `x1`, `x2`, `x3`, and `y` must have 2D logical shapes, with `x3.shape == y.shape == (M, N)`.
- Batch dimensions (including size-1 dimensions) and broadcasting are not supported. The 4D physical storage of NZ weights does not count as batch dimensions.
- If `bias` is supplied, its shape must be `(N,)`.

## Invocation Description

This operator does not support GE graph mode.

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn API | [FP8 (sigmoid + mul) example](examples/arch35/test_aclnn_quant_matmul_sigmoid_mx_quant.cpp), [FP8 (GELU) example](examples/arch35/test_aclnn_quant_matmul_gelu_mx_quant.cpp), [FP8 (SwiGLU) example](examples/arch35/test_aclnn_quant_matmul_swiglu_mx_quant.cpp) | Calls [aclnnQuantMatmulActivationQuantV2](docs/aclnnQuantMatmulActivationQuantV2_en.md) with x2 in ND format. |
| aclnn API | [FP8 (sigmoid + mul) example with NZ weights](examples/arch35/test_aclnn_quant_matmul_sigmoid_mx_quant_weight_nz.cpp), [FP8 (GELU) example with NZ weights](examples/arch35/test_aclnn_quant_matmul_gelu_mx_quant_weight_nz.cpp) | Calls [aclnnQuantMatmulActivationQuantWeightNzV2](docs/aclnnQuantMatmulActivationQuantWeightNzV2_en.md) with x2 in FRACTAL_NZ format. |
| aclnn API | [Example](docs/aclnnQuantMatmulActivationQuant_en.md#example) | Calls [aclnnQuantMatmulActivationQuant](docs/aclnnQuantMatmulActivationQuant_en.md) with x2 in ND format. |
| aclnn API | [Example](docs/aclnnQuantMatmulActivationQuantWeightNz_en.md#example) | Calls [aclnnQuantMatmulActivationQuantWeightNz](docs/aclnnQuantMatmulActivationQuantWeightNz_en.md) with x2 in FRACTAL_NZ format. |
| PyTorch API (eager mode) | [FP8 and FP4 examples](docs/torchapi_quant_matmul_activation_quant_en.md#example) | Calls [cann_ops_nn.quant_matmul_activation_quant](docs/torchapi_quant_matmul_activation_quant_en.md). For extension installation, see the [PyTorch extension instructions](../../torch_extension/README_en.md). |

## References

- For test execution instructions, see the [test guide](tests/README_en.md).
