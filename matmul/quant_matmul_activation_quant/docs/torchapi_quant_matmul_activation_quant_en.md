# quant_matmul_activation_quant

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

- Description: This API fuses MX (Microscaling) quantized matrix multiplication, activation, and dynamic MX quantization to return the quantized result `y` and its group scales `y_scale`. It supports GELU (Gaussian Error Linear Unit, tanh/erf forms), SiLU (Sigmoid Linear Unit), SwiGLU (Swish Gated Linear Unit), and sigmoid followed by elementwise multiplication with `x3`.

  The extension calls [aclnnQuantMatmulActivationQuantV2](aclnnQuantMatmulActivationQuantV2_en.md) for ND weights and [aclnnQuantMatmulActivationQuantWeightNzV2](aclnnQuantMatmulActivationQuantWeightNzV2_en.md) for FRACTAL_NZ weights, including calls that omit `x3`.

- Computation: Perform MX matrix multiplication and add the bias, apply the activation, and then dynamically quantize the result in MX mode.

  Batch indices are omitted below. $X_1$ and $X_2$ are the matrices used in the computation, with shapes $(M,K)$ and $(K,N)$, respectively. M is the row count of the left matrix, K is the shared reduction dimension, and N is the column count of the right matrix. Transposition is expressed through tensor views; see [Layout and Storage](#layout-and-storage).

### MX Quantized Matrix Multiplication

Split K into groups of 32 elements. Multiply each partial matrix product by the corresponding input scales and accumulate the results with bias to obtain $C$:

$$
C_{m,n}=\sum_{j=0}^{L-1}\left[
\left(\sum_{t=0}^{g_j-1}X_{1,m,jg_K+t}\,X_{2,jg_K+t,n}\right)
S_{1,m,j}\,S_{2,j,n}\right]+b_n
$$

Here, $g_M$, $g_N$, and $g_K$ are the group sizes along M, N, and K, respectively, and are currently fixed at $(1,1,32)$; $L=\lceil K/g_K\rceil$ and $g_j=\min(g_K,K-jg_K)$. $S_1$ and $S_2$ denote the scale values for group $j$ in `x1_scale` and `x2_scale`, respectively. For storage shapes, see [Layout and Storage](#layout-and-storage). Matrix multiplication uses FP32 accumulation, and bias is added before activation. If bias is omitted, $b_n=0$.

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

- `activation_type="sigmoid"` applies sigmoid to $C$ and multiplies it elementwise by the BF16 tensor $B$ supplied in `x3`.

  $$
  A=\operatorname{sigmoid}(C)\odot\operatorname{FP32}(B)
  $$

  Sigmoid and multiplication use the FP32 accumulation result. The product is rounded once to BF16, using round-to-nearest, ties-to-even, before dynamic MX quantization.

All activations above are calculated in FP32. Let $H$ be the output width after activation: $H=N$ for GELU, SiLU, and sigmoid, and $H=N/2$ for SwiGLU.

### Dynamic MX Quantization

Convert the activation result to BF16 using Round to Nearest, ties to Even (RNE): round to the nearest representable value, resolving ties to the value whose least significant mantissa bit is 0. Then divide the last dimension into groups of 32 elements. If the last group contains fewer than 32 elements, pad it with zeros for scale calculation. The padding elements are not written to the output.

$$
V_i=\operatorname{BF16}_{RNE}(A_i),\qquad a=\max_i|V_i|,\qquad s=2^e
$$

Each group shares a scale $s$. Convert the normalized values to output type $D$ using `round_mode`:

$$
Q_i=\begin{cases}
\operatorname{FP32}(V_i/s), & D\text{ is FP8}\\
\operatorname{BF16}_{RNE}(V_i/s), & D\text{ is FP4}
\end{cases},\qquad y_i=\operatorname{Cast}_{D,\,\mathrm{round\_mode}}(Q_i)
$$

`y_scale` uses `torch.float8_e8m0fnu`. The scale for group $j$ is written to `y_scale[..., m, j // 2, j % 2]`; its shape is `(..., M, ceil(H/64), 2)`.

For the formulas and special-value handling of each `scale_alg`, see [Scale Calculation Algorithms](../README_en.md#scale-calculation-algorithms).

## Function Prototype

Call `cann_ops_nn.quant_matmul_activation_quant`:

```python
cann_ops_nn.quant_matmul_activation_quant(
    x1,
    x2,
    x2_scale,
    *,
    x1_scale=None,
    bias=None,
    output_dtype=None,
    x1_dtype=None,
    x2_dtype=None,
    x1scale_dtype=None,
    x2scale_dtype=None,
    group_sizes=None,
    activation_type="gelu_tanh",
    quant_mode="mx",
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0.0,
    x3=None,
) -> Tuple[Tensor, Tensor]
```

## Parameters

All tensors must be nonempty and reside on the same NPU device. The table gives logical shapes. FP8 matrices can use their native tensor types. For FP4, the table shows `torch.uint8` storage with two E2M1 values per byte. See [Layout and Storage](#layout-and-storage) for packed tensor shapes.

`batch1` and `batch2` denote the zero to four leading batch dimensions of each input. `batch` is the result of right-aligned broadcasting. Omit the batch dimensions when absent. `ceil` rounds up to the nearest integer.

| Parameter | Parameter Type | Optional/Required | Description | Data Type | Dimensions (Shape) |
| --- | --- | --- | --- | --- | --- |
| x1 | Tensor | Required | Left matrix, corresponding to $X_1$. Uses ND. | torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8 (packed FP4) | 2–6 dimensions, `(batch1,M,K)` |
| x2 | Tensor | Required | Right matrix, corresponding to $X_2$. Uses ND or FRACTAL_NZ. NZ does not support torch.float8_e5m2. | torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8 (packed FP4) | 2–6 dimensions, `(batch2,K,N)` |
| x2_scale | Tensor | Required | Group scales for x2, corresponding to $S_2$. Uses ND. | torch.float8_e8m0fnu | 3–7 dimensions, `(batch2,ceil(K/64),N,2)` |
| x1_scale | Tensor | Optional | Group scales for x1, corresponding to $S_1$. Uses ND. Defaults to None; this API supports only MX mode, so a non-None tensor must be supplied when calling it. | torch.float8_e8m0fnu | 3–7 dimensions, `(batch1,M,ceil(K/64),2)` |
| bias | Tensor | Optional | Bias added before activation, corresponding to $b$. Uses ND. Defaults to None (no bias). Supports `(N,)`; `(B,1,N)` is supported only when y is 3D, with B equal to the broadcast batch size. For SwiGLU, the bias length remains the pre-activation width N. | torch.float32 | `(N,)` or `(B,1,N)` |
| output_dtype | int | Optional | Logical data type identifier for y. Defaults to None, which selects the logical type of x1. Accepts the same values as x1_dtype, and the specified type must match the logical type of x1. | - | - |
| x1_dtype | int | Optional | Logical data type identifier for x1. Defaults to None (inferred from x1.dtype). FP8 can be specified explicitly with torch_npu.float8_e5m2 or torch_npu.float8_e4m3fn; FP4 requires torch_npu.float4_e2m1fn_x2. | - | - |
| x2_dtype | int | Optional | Logical data type identifier for x2. Defaults to None (inferred from x2.dtype). Accepts the same values as x1_dtype, except that NZ does not support torch_npu.float8_e5m2; FP4 requires torch_npu.float4_e2m1fn_x2. | - | - |
| x1scale_dtype | int | Optional | Logical data type identifier for x1_scale. Defaults to None (inferred from x1_scale.dtype). Use torch_npu.float8_e8m0fnu to specify it explicitly. | - | - |
| x2scale_dtype | int | Optional | Logical data type identifier for x2_scale. Defaults to None (inferred from x2_scale.dtype). Use torch_npu.float8_e8m0fnu to specify it explicitly. | - | - |
| group_sizes | List[int] | Optional | Input quantization group sizes along M, N, and K, ordered as `[groupSizeM,groupSizeN,groupSizeK]`. Defaults to None. Only None, an empty list, `[0,0,0]`, or `[1,1,32]` is supported; all select `[1,1,32]`. | - | - |
| activation_type | str | Optional | Activation type. Supports "gelu_tanh", "gelu_erf", "silu", "swiglu", and "sigmoid". Defaults to "gelu_tanh". | - | - |
| quant_mode | str | Optional | Quantization mode. Only "mx" is supported. Defaults to "mx". | - | - |
| round_mode | str | Optional | Rounding mode when converting to the output data type. Defaults to "rint": round to the nearest representable value, with ties to even. FP8 supports only this mode. FP4 also supports "floor" (round toward negative infinity) and "round" (round to the nearest representable value, with ties away from zero). | - | - |
| scale_alg | int | Optional | Algorithm for calculating output scales: 0 uses the OCP (Open Compute Project) shared exponent; 1 uses the FP8 maximum finite value; 2 uses the range specified by dst_type_max. FP8 supports 0/1; FP4 supports 0/2. Defaults to 0; None also selects 0. | - | - |
| dst_type_max | float | Optional | Reference maximum used to calculate output scales when scale_alg=2, corresponding to d in the quantization formulas. Defaults to 0.0, which selects 6.0; finite values in [6.0,12.0] are also supported. Ignored by the other algorithms. | - | - |
| x3 | Tensor | Optional | ND tensor $B$ multiplied elementwise with the sigmoid result. Defaults to `None`. Supply it for `activation_type="sigmoid"` and omit it for other activations. See [Sigmoid Scenario](#sigmoid-scenario). | torch.bfloat16 | `(M, N)` |

The `torch_npu` constants used for type parameters are integer type identifiers. With native FP8 or E8M0 tensors, the corresponding parameters can remain None. An explicit input type changes how the data is interpreted; it does not perform a numeric conversion. The data must already use the corresponding representation.

## Return Values

Returns two contiguous tensors, `(y, y_scale)`, on the same NPU device as `x1`. The table gives the actual dtype and shape of each returned tensor. The logical output width $H$ equals N for GELU, SiLU, and sigmoid, and N/2 for SwiGLU.

| Parameter | Parameter Type | Optional/Required | Description | Data Type | Dimensions (Shape) |
| --- | --- | --- | --- | --- | --- |
| y | Tensor | Required | Dynamically quantized result $y$. Its logical type matches `x1`. In FP4 mode each byte holds two E2M1 output values. | torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8 (packed FP4) | `(batch, M, H)` for FP8, `(batch, M, H/2)` for FP4 |
| y_scale | Tensor | Required | One scale per 32 output elements, corresponding to $s$; two consecutive scales occupy the trailing dimension of size 2. | torch.float8_e8m0fnu | `(batch,M,ceil(H/64),2)` |

## Constraints

- This API performs quantized forward computation. The extension does not currently provide backward gradients for this operator.
- This API supports single-operator invocation. The extension does not currently provide a TorchAir graph conversion adapter for this operator.

### Data Types and Shapes

- x1 and x2 have 2 to 6 logical dimensions. K must match, and batch dimensions use right-aligned broadcasting. Each input scale's batch dimensions must exactly match those of its matrix.
- Each logical dimension of every input and output tensor, including scales and any supplied optional tensors, must be in the range $[1,2^{32}-1]$.
- Storage dimensions must be positive. For each tensor, multiply the dimensions of its logical shape and its storage shape separately. Each element count must be at most $2^{63}-1$. The NZ storage count includes alignment padding. Count FP4 data in 4-bit elements.
- The product of the broadcast batch dimensions, `batchSize`, must not exceed $2^{32}-1$. This product excludes M, K, and N; `batchSize` is 1 when there are no batch dimensions.
- The number of output elements before activation, `batchSize * M * N`, must not exceed $2^{63}-1$, where N is the number of columns in the matrix multiplication output before activation.
- Both input scales and the output scales use torch.float8_e8m0fnu.

Supported logical data type combinations are listed below.

| x1 | x2 | y | x2 format |
| --- | --- | --- | --- |
| torch.float8_e4m3fn | torch.float8_e4m3fn | torch.float8_e4m3fn | ND, FRACTAL_NZ |
| torch.float8_e5m2 | torch.float8_e4m3fn | torch.float8_e5m2 | ND, FRACTAL_NZ |
| torch.float8_e4m3fn | torch.float8_e5m2 | torch.float8_e4m3fn | ND |
| torch.float8_e5m2 | torch.float8_e5m2 | torch.float8_e5m2 | ND |
| FP4 (E2M1) | FP4 (E2M1) | FP4 (E2M1) | ND, FRACTAL_NZ |

In the following table, FP8 denotes E4M3FN or E5M2 and FP4 denotes E2M1. Use the type identifier `torch_npu.float4_e2m1fn_x2` for packed FP4 input. The matrix types must also satisfy the combinations above.

| activation_type | Data Type | N Requirement | x1 Layout When x2 Is ND | x1 Layout When x2 Is NZ |
| --- | --- | --- | --- | --- |
| "gelu_tanh", "gelu_erf" | FP8 | ND $N>0$, NZ $N>1$ | Transpose views supported | Transpose views supported |
| "gelu_tanh", "gelu_erf" | FP4 | Positive even integer | Transpose views supported | Transpose views not supported |
| "silu" | FP8 | ND $N>0$, NZ $N>1$ | Transpose views supported | Transpose views not supported |
| "sigmoid" | FP8 | ND $N>0$, NZ $N>1$ | Transpose views supported | Transpose views not supported |
| "swiglu" | FP8 | Positive multiple of 64 | Transpose views supported | Transpose views not supported |

All the combinations above support transpose views of `x2`. NZ weights also require $K>1$. FP4 supports GELU only and requires $K>2$ and even N. The innermost dimension of the contiguous FP4 input before transposition must be even. For `x1` this is K for a contiguous input and M for a transpose view. For `x2` it is N for a contiguous input and K for a transpose view. With NZ weights, both K and N must be even.

### Sigmoid Scenario

For `activation_type="sigmoid"`:

- `x1`, `x2`, `x3`, and `y` must have 2D logical shapes, with `x3.shape == y.shape == (M, N)`.
- Batch dimensions (including size-1 dimensions) and broadcasting are not supported. The 4D physical storage of NZ weights does not count as batch dimensions.
- If `bias` is supplied, its shape must be `(N,)`.

### Layout and Storage

This API expresses transposition through tensor views. FP8 inputs are always passed with view shapes `x1=(batch1,M,K)` and `x2=(batch2,K,N)`. Scale shapes retain the axis order listed in the parameter table.

For contiguous layouts and standard transpose views, each matrix and its scale must use matching view orientations. If either scale axis being exchanged has size 1, the equivalent shape with these two axes swapped is also accepted. The swapped shape must match the required full shape, including the batch dimensions and the final dimension of 2. For NZ weights, convert the contiguous matrix to NZ before creating the transpose view.

For example, if x1 is stored as `(K,M)`, transpose the matrix's last two axes and the scale's third-last and second-last axes:

```python
# x1_base.shape=(K,M), x1_scale_base.shape=(ceil(K/64),M,2)
x1 = x1_base.transpose(-1, -2)
x1_scale = x1_scale_base.transpose(-3, -2)
```

The following uses `torch.uint8` to illustrate the packed FP4 input layout. Each byte contains two FP4 elements: the low four bits encode the first element, and the high four bits encode the second. Along the packed axis, each tensor element represents two logical elements, so the tensor dimension is half the logical dimension. The input byte layout, shape, strides, and storage offset must follow this convention.

Set both `x1_dtype` and `x2_dtype` to `torch_npu.float4_e2m1fn_x2`; `output_dtype` may be omitted or set to the same constant. The tensor shapes for this packed layout are listed below, with batch dimensions omitted:

| Input | Contiguous Layout | Transpose View |
| --- | --- | --- |
| x1 | `(M,K/2)` | Transpose `(K,M/2)` to obtain `(M/2,K)` |
| x2 | `(K,N/2)` | Transpose `(N,K/2)` to obtain `(K/2,N)` |

Calculate scale shapes using logical M, N, and K. Each group of 32 K elements has one input scale; scales for two consecutive groups occupy the trailing dimension of size 2. Scales use E8M0 and are not packed as FP4.

ND inputs support [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md). Strides and storage offsets must be nonnegative, and all accesses must stay within the underlying storage. NZ weights support only complete matrices and views transposing their last two dimensions; slices selecting part of the NZ storage are not supported, and `storage_offset()` must be 0. Use `torch_npu.npu_format_cast` for NZ format conversion, as shown in the example below.

## Determinism and Batch Consistency

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>:
  - Deterministic computation: Supported by default.
  - Batch consistency: [Batch consistency](../../../docs/en/context/batch_consistency.md) is supported by default under the following conditions. In the same runtime environment, keep N, K, data types, layouts, and activation/quantization attributes unchanged. For a given token, keep its x1 and x1_scale data and the corresponding x2, x2_scale, and bias data unchanged. For `activation_type="sigmoid"`, also keep the corresponding x3 row data and its token pairing with x1 unchanged. Under these conditions, changing only the batch size, M, or the positions or combinations of tokens preserves bitwise-identical y and y_scale values for that token. No additional switch is required.
<!-- end id7 -->

## Example

Install compatible `torch`, `torch_npu`, and the repository's PyTorch extension. See the [PyTorch Extension Guide](../../../torch_extension/README_en.md) for installation instructions.

### Single-Operator Invocation

All examples use MX quantization and are organized by matrix data type and activation combination.

#### FP8 (SwiGLU)

This example uses SwiGLU and NZ weights. Set `activation_type` to `"gelu_tanh"`, `"gelu_erf"`, or `"silu"` for other activations. To use ND weights, omit the `npu_format_cast` line.

```python
import torch
import torch_npu
import cann_ops_nn

m, k, n = 5, 64, 128
activation_type = "swiglu"
x1 = torch.randn(m, k).to(torch.float8_e4m3fn).npu()
x2 = torch.randn(k, n).to(torch.float8_e4m3fn).npu()
x2 = torch_npu.npu_format_cast(x2, 29)  # FRACTAL_NZ

scale_k = (k + 63) // 64
# E8M0 byte 127 encodes scale=1.
x1_scale = torch.full((m, scale_k, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
x2_scale = torch.full((scale_k, n, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
bias = torch.zeros(n, dtype=torch.float32).npu()

y, y_scale = cann_ops_nn.quant_matmul_activation_quant(
    x1,
    x2,
    x2_scale,
    x1_scale=x1_scale,
    bias=bias,
    activation_type=activation_type,
    scale_alg=1,
)
h = n // 2 if activation_type == "swiglu" else n
assert tuple(y.shape) == (m, h)
assert tuple(y_scale.shape) == (m, (h + 63) // 64, 2)
print(y.cpu(), y_scale.cpu())
```

#### FP8 (sigmoid + mul)

This example sets `activation_type="sigmoid"` and uses E4M3FN matrices, E8M0 input scales, and a BF16 ND tensor `x3`. It demonstrates both ND and NZ weights. `x3` must have the same shape as the output, `(5,128)` in this example. Sigmoid preserves the output width.

Matrix multiplication and sigmoid are computed in FP32. After multiplication by `x3`, the product is rounded to BF16 with RINT before MX quantization. For other activations, omit `x3`.

```python
import torch
import torch_npu
import cann_ops_nn

m, k, n = 5, 64, 128
x1 = torch.full((m, k), 0.125).to(torch.float8_e4m3fn).npu()
x2 = torch.full((k, n), 0.125).to(torch.float8_e4m3fn).npu()
scale_k = (k + 63) // 64
# E8M0 byte 127 encodes scale=1; x3 is a separate BF16 ND tensor.
x1_scale = torch.full((m, scale_k, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
x2_scale = torch.full((scale_k, n, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
x3 = torch.full((m, n), 0.5, dtype=torch.bfloat16).npu()

for use_nz in (False, True):
    # Convert only weight x2; input scales and x3 remain in ND format.
    weight = torch_npu.npu_format_cast(x2, 29) if use_nz else x2
    y, y_scale = cann_ops_nn.quant_matmul_activation_quant(
        x1,
        weight,
        x2_scale,
        x1_scale=x1_scale,
        x3=x3,
        activation_type="sigmoid",
        round_mode="rint",
        scale_alg=0,
    )
    assert y.dtype == torch.float8_e4m3fn
    assert tuple(y.shape) == (m, n)
    assert tuple(y_scale.shape) == (m, (n + 63) // 64, 2)
    print("NZ" if use_nz else "ND", y.cpu(), y_scale.cpu())
```

#### FP4 (GELU)

This example uses ND weights and GELU, with packed FP4 inputs stored in torch.uint8. The current extension also returns FP4 output y packed in torch.uint8; the scale shape is calculated from the logical output width N.

```python
import torch
import torch_npu
import cann_ops_nn

m, k, n = 5, 64, 128
# 0x22 encodes two FP4 (E2M1) elements, both equal to 1.
x1 = torch.full((m, k // 2), 0x22, dtype=torch.uint8).npu()
x2 = torch.full((k, n // 2), 0x22, dtype=torch.uint8).npu()
scale_k = (k + 63) // 64
x1_scale = torch.full((m, scale_k, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
x2_scale = torch.full((scale_k, n, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()

y, y_scale = cann_ops_nn.quant_matmul_activation_quant(
    x1,
    x2,
    x2_scale,
    x1_scale=x1_scale,
    x1_dtype=torch_npu.float4_e2m1fn_x2,
    x2_dtype=torch_npu.float4_e2m1fn_x2,
    output_dtype=torch_npu.float4_e2m1fn_x2,
    activation_type="gelu_tanh",
    scale_alg=2,
    dst_type_max=6.0,
)
assert y.dtype == torch.uint8
assert y_scale.dtype == torch.float8_e8m0fnu
assert tuple(y.shape) == (m, n // 2)
assert tuple(y_scale.shape) == (m, (n + 63) // 64, 2)
print(y.cpu(), y_scale.cpu())
```
