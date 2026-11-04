# SwigluGroupQuantWithDualAxis

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

### Description

This operator fuses Clipped-SwiGLU, optional row-wise weight multiplication, and MX FP8 quantization along **two directions**. A single invocation outputs five results:

- `y1` and `mxScale1`: quantized along the last dimension (axis -1); every 32 elements within each token row share one scale.
- `y2` and `mxScale2`: quantized along the row direction (axis -2); in the group scenario, according to the grouping boundaries given by `group_index`, every 32 rows and each column within a group share one scale; in the non-group scenario, all rows form blocks.
- `yOrigin`: the optional activation result before weight multiplication.

The quantized values of the first path are **byte-by-byte identical** to those of `quantMode=5` of the single-axis operator `SwigluGroupQuant` (both reuse the same MX kernel). Therefore, the first-path results of the training side (this operator) and the inference side (single-axis `quantMode=5`) are aligned.

### Formula

The Clipped-SwiGLU activation can be written uniformly as:

$$
A'[t,h] =
\begin{cases}
\min(A[t,h], L), & L > 0 \\
A[t,h], & L = -1
\end{cases}
$$

$$
B'[t,h] =
\begin{cases}
\min(\max(B[t,h], -L), L), & L > 0 \\
B[t,h], & L = -1
\end{cases}
$$

$$
F[t, h] = A'[t,h] \times \sigma\!\left(\alpha \times A'[t,h]\right)
\times \left(B'[t,h] + \beta\right)
$$

where:

- $A = x[:, :H]$ and $B = x[:, H:]$, with $H = D/2$
- $L$ is `clampLimit`; only `-1.0` (Clamp disabled) or a finite positive number is supported
- $\alpha$ is `alpha`, and $\beta$ is `bias`
- $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ is the Sigmoid function

**Basic computation process**

```txt
Step 1: Input splitting (split along the last dimension into A and B)
Step 2: Clamp processing (optional)
Step 3: Variant SwiGLU activation
Step 4: Weight multiplication (optional)
Step 5: Dual-axis MX FP8 quantization (first path: axis -1 / second path: axis -2)
```

<details>
<summary><strong>Step 1: Input splitting</strong></summary>

The input tensor $\mathbf{x} \in \mathbb{R}^{T \times D}$ is split along the last dimension into two parts ($H = D/2$):

$$
A[t, h] = \mathbf{x}[t, h], \quad h \in [0, H)
$$

$$
B[t, h] = \mathbf{x}[t, h + H], \quad h \in [0, H)
$$

</details>

<details>
<summary><strong>Step 2: Clamp processing (optional)</strong></summary>

When `clampLimit > 0`:

$$
A'[t, h] = \min(A[t, h], L)
$$

$$
B'[t, h] = \min(\max(B[t, h], -L), L)
$$

When `clampLimit = -1.0`, Clamp is not enabled, and $A'=A$, $B'=B$.

**Clamp semantics**: the gating branch $A$ is truncated only by an upper bound, that is, $A' \in (-\infty, L]$; the linear branch $B$ is truncated symmetrically, that is, $B' \in [-L, L]$.

</details>

<details>
<summary><strong>Step 3: Variant SwiGLU activation</strong></summary>

$$
F[t, h] = A'[t, h] \cdot \sigma\!\left(\alpha \cdot A'[t, h]\right) \cdot \left(B'[t, h] + \beta\right)
$$

When $\alpha = 1.0$, $\beta = 0.0$, and `clampLimit = -1.0` (Clamp disabled), the computation degenerates to standard SwiGLU. The activation is computed in FP32 precision, and the final result is rounded to the input dtype (denoted as $R_T(\cdot)$):

$$
\text{yOrigin} = R_T(F)
$$

When `output_origin=true`, this value is output (excluding the weight influence) and can be directly used for backward computation of `grad_weight`.

</details>

<details>
<summary><strong>Step 4: Weight multiplication (optional)</strong></summary>

When `weight` is provided:

$$
u[t, h] = R_T\!\left(R_T(F[t, h]) \cdot w[t]\right)
$$

When `weight` is absent, $u = R_T(F)$. $u$ is the quantization input shared by the two quantization branches and is materialized in `x.dtype` (FP16/BF16).

</details>

<details>
<summary><strong>Step 5: Dual-axis MX FP8 quantization</strong></summary>

Both quantization paths follow the CuBALS (`scaleAlg=1`) scale algorithm, with a fixed block size of 32.
Before quantization, the maximum absolute value within a block is computed first, and then mapped to the representable range of the target FP8 type:

$$
s^{raw} = \frac{amax}{Amax(DType)}
$$

where $Amax(DType)$: $448.0$ for FP8_E4M3FN, and $57344.0$ for FP8_E5M2.

For a finite and nonzero $s^{raw}$, interpret it according to the FP32 bit pattern. Let $E_b$ denote the **8 bit biased exponent field** of FP32 (biased exponent),
and let $m \in [0,1)$ denote the fractional part corresponding to the mantissa field. CuBALS rounds the exponent up:

$$
E_b^{*} =
\begin{cases}
E_b + 1, & 0 < E_b < 254 \text{ and } m > 0 \\
E_b + 1, & E_b = 0 \text{ and } m > 0.5 \\
E_b, & \text{otherwise}
\end{cases}
$$

E8M0 directly stores this biased exponent encoding:

$$
mxScale = E_b^{*}, \qquad s = 2^{E_b^{*}-127}
$$

Data within a block is quantized as $d = Q8(u / s)$; FP32→FP8 uses **RINT (round to nearest)**.
If $amax = 0$ within a block (all-zero block), then `mxScale = 0`; unused scale padding/reserved positions are not part of valid results, and callers should not rely on their values.

<details>
<summary><strong>First path (axis -1, `y1` / `mxScale1`)</strong></summary>

Within each token row, every 32 elements along the last dimension form a block, and the scale is computed independently for each row:

$$
amax[t, c] = \max_{i=0}^{31} |u[t, 32c + i]|
$$

$$
mxScale1[t, c] = \text{CuBALS}(amax[t, c]), \qquad y1[t, j] = Q8\!\left(\frac{u[t, j]}{s[t, \lfloor j/32 \rfloor]}\right)
$$

The physical layout of `mxScale1` is `[T, ceil(ceil(H/32)/2), 2]`; the last dimension stores at most 2 adjacent 32-element block scales of the same row. If the last pair is not filled, the unused positions carry no guarantee of valid values, and callers should not read or rely on their content.

</details>

<details>
<summary><strong>Second path (axis -2, `y2` / `mxScale2`)</strong></summary>

**Group scenario**: according to the grouping boundaries given by `group_index`, every 32 rows within a group form a block, and the same column shares one scale;
blocks do not cross groups. When the end of a group has fewer than 32 rows, zero padding is applied only within the temporary block for scale computation, and no additional valid rows of `y2` are produced:

$$
amax[g, b, j] = \max_{0 \le i < \min(32,\ e_g-e_{g-1}-32b)} |u[e_{g-1} + 32b + i,\ j]|
$$

$$
y2[t, j] = Q8\!\left(\frac{u[t, j]}{s\big[g(t),\ \lfloor (t - e_{g(t)-1})/32 \rfloor,\ j\big]}\right)
$$

**Non-group scenario**: it degenerates to all rows forming blocks of 32 rows, with the scale computed independently for each column.

The physical shape of `mxScale2` is `[floor(T/64) + G, H, 2]` in the group scenario and `[ceil(T/64), H, 2]` in the non-group scenario.
`slot0/slot1` in the last dimension correspond to at most two adjacent 32-row block scales. In the group scenario, additional physical positions are reserved to keep grouping boundaries;
empty groups, the tail with an odd number of 32-row blocks, and other unused reserved positions carry no guarantee of valid values, and callers should not read or rely on their content.

</details>

</details>

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 130px">
  <col style="width: 130px">
  <col style="width: 430px">
  <col style="width: 190px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>SWIGLU input. The last dimension is evenly split into parts A and B. The shape is [T, 2H], where 2H is greater than or equal to 64 and divisible by 64. Non-contiguous tensors are supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input (optional)</td>
      <td>Per-token weight. When it is not empty, it is multiplied onto the activation result. It is supported only in the group scenario, where group_index must be provided at the same time. The number of dimensions is 1 to 8, and the number of elements must equal T. Non-contiguous tensors are supported.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>Input (optional)</td>
      <td>Cumulative group-end index (with cumsum semantics; each element value itself is the group-end row index). It is used only for the grouped quantization of the second path (axis -2) and does not perform output truncation. The shape is 1D [G], and it is not empty; elements are non-negative and non-decreasing, the last element equals T, and adjacent equal elements indicate empty groups. If it is not passed, the non-group scenario applies. Non-contiguous tensors are supported.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Attribute</td>
      <td>Target quantization type. The ACLNN/GE encoding supports 35 and 36, which represent FLOAT8_E5M2 and FLOAT8_E4M3FN respectively; the Torch encoding supports 291 (float8_e5m2) and 292 (float8_e4m3fn), with 292 as the default. The implementation is compatible with both 35 and 36.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Attribute</td>
      <td>Quantization mode. Only 1 is supported, which indicates dual-axis MX FP8 quantization.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clamp_limit</td>
      <td>Attribute</td>
      <td>Clamp threshold of Clipped-SwiGLU. The default -1.0 indicates that Clamp is not enabled; when it is enabled, the value must be a finite positive number.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_origin</td>
      <td>Attribute</td>
      <td>Whether to output the activation yOrigin before weight multiplication. Default: false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Attribute</td>
      <td>Scaling coefficient of the sigmoid input in the variant SwiGLU. Default: 1.0. The value must be a finite positive number.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Attribute</td>
      <td>Bias of the linear branch in the variant SwiGLU. Default: 0.0. The value must be a finite number.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>Quantized output of the first path (axis -1). The shape is [T, H].</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxScale1</td>
      <td>Output</td>
      <td>Quantization scale of the first path. The shape is [T, ceil(ceil(H/32)/2), 2]. The last dimension stores at most 2 adjacent 32-element block scales of the same row; unused padding positions carry no guarantee of valid values, and callers should not rely on their content.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Quantized output of the second path (axis -2). The shape is [T, H].</td>
      <td>FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxScale2</td>
      <td>Output</td>
      <td>Quantization scale of the second path. The shape is [floor(T/64) + G, H, 2] in the group scenario and [ceil(T/64), H, 2] in the non-group scenario, where G is the number of elements of group_index. slot0/slot1 in the last dimension correspond to at most two adjacent 32-row block scales; the group scenario may contain physical positions reserved for grouping boundaries, and unused reserved slots carry no guarantee of valid values, so callers should not rely on their content.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>yOrigin</td>
      <td>Output</td>
      <td>Activation result before weight multiplication. When output_origin is true, the shape is [T, H] and the data type is the same as that of x; when it is false, the shape is a [0] placeholder. Null pointers are not supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- `x` supports only 2D `[T, 2H]`, FLOAT16 or BFLOAT16, with `2H >= 64` and divisible by 64.
- `quant_mode` supports only `1`; `dst_type` supports only FLOAT8_E4M3FN (default) and FLOAT8_E5M2.
- `weight` is allowed only when `group_index` is provided, and the number of elements must equal `T`; passing `weight` in the non-group scenario will be rejected.
- `group_index` is the cumulative group-end index, and element values are validated during **device-side** execution: invalid values (negative values, decreasing values, out-of-bounds values, or a last element not equal to `T`)
  trigger a device execution exception, and an asynchronous invocation may report the failure only at a subsequent stream synchronization; after a failure, all outputs are unusable.
  The `GetWorkspaceSize` phase validates only the dtype, number of dimensions, and number of elements, and does not guarantee that element values are valid.
- `clamp_limit = -1.0` disables Clamp; otherwise, the value must be a finite positive number; `alpha` must be a finite positive number, and `bias` must be finite.
- Forward only; autograd is not supported.
- Deterministic computation is supported by default.
- `x`, `weight`, and `group_index` all support non-contiguous tensors (automatically converted to contiguous internally).
- In the outputs, `H = D/2 = x.shape[-1]/2` and `G = group_index.numel()`; unused padding or reserved positions in `mxScale1` and `mxScale2` carry no guarantee of valid values, and callers should not rely on their content.

## Invocation Description

| Invocation method | Invocation sample | Description |
|:-------|:-------|:---|
| aclnn invocation | [test_aclnn_swiglu_group_quant_with_dual_axis](./examples/test_aclnn_swiglu_group_quant_with_dual_axis.cpp) | Invokes the SwigluGroupQuantWithDualAxis operator through the [aclnnSwigluGroupQuantWithDualAxis](./docs/aclnnSwigluGroupQuantWithDualAxis.md) interface. |
| Graph-mode invocation | - | Invokes the SwigluGroupQuantWithDualAxis operator by building a graph through the [operator IR](./op_graph/swiglu_group_quant_with_dual_axis_proto.h). |
