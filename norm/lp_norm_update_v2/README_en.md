# LpNormUpdateV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: LpNormUpdateV2 is the "update" (Update) phase operator of the two-step Lp norm computation. It receives the reduction result of LpNormReduceV2, `x = Σ|x|^p` (a non-negative real number), selects the corresponding root-extraction branch based on the norm order `p` to compute `f_p(x)`, takes the maximum with the numerical stability constant `epsilon`, and obtains the update result `y = max(f_p(x), ε_eff)`.

- Formula:

  $$
  y = \max\left(f_p(x),\ \varepsilon_{\text{eff}}\right)
  $$

  Here, $f_p(x)$ has 3 branches based on the value of $p$:

  | Branch | Trigger Condition | Formula |
  |------|---------|---------|
  | Identity branch | p ∈ {0, 1, ±inf} | f_p(x) = x |
  | sqrt branch | p ≈ 2.0 | f_p(x) = √x |
  | Power operation branch | other p values | f_p(x) = exp(ln(x)/p) |

  The effective epsilon $\varepsilon_{\text{eff}}$ is affected by the input dtype: for FP16, when $\varepsilon \leq 10^{-7}$, it is raised to $10^{-7}$ (to prevent underflow). For FP32/BF16, $\varepsilon$ is used directly.

- Usage scenario: serves as a key substep of `torch.linalg.vector_norm` on the Ascend NPU and is used together with LpNormReduceV2. Typical scenarios include gradient clipping (`clip_grad_norm_`) and weight normalization (WeightNorm) during training.

  End-to-end equivalence relationship:

  ```text
  LpNormReduceV2(x, p) → LpNormUpdateV2(S, p, epsilon=0) ≡ linalg_vector_norm(x, ord=p)
  ```

## Parameters

<table style="table-layout: fixed"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>The reduction result of LpNormReduceV2, Σ|input|^p, a non-negative real number.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>p</td>
    <td>Attribute</td>
    <td><ul><li>The norm order. The default value is 2.0.</li><li>Any floating-point value and ±inf are supported.</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>Attribute</td>
    <td><ul><li>The numerical stability constant. The default value is 1e-12.</li><li>For FP16, when ε≤1e-7, it is automatically raised to 1e-7.</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>The update result max(f_p(x), ε_eff), with the same shape and dtype as x.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

GE IR definition:

```cpp
REG_OP(LpNormUpdateV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(p, Float, 2.0)
    .ATTR(epsilon, Float, 1e-12)
    .OP_END_FACTORY_REG(LpNormUpdateV2)
```

## Constraints

- The input `x` must be a non-negative real number (the reduction result of LpNormReduceV2, Σ|input|^p ≥ 0). Otherwise, the log branch produces NaN.
- The output `y` has exactly the same shape and dtype as the input `x`, with no broadcasting (element-wise operation).
- dtype precision promotion strategy:
  - FP16: intermediate computation is promoted to FP32 (FP16→FP32→FP16), aligning with PyTorch `opmath_type<Half>=float`.
  - BF16: intermediate computation is promoted to FP32 (BF16→FP32→BF16), and the output is restored through rounding.
  - FP32: computation is performed directly at FP32 precision.
- epsilon protection: for FP16, when ε≤1e-7, it is automatically raised to 1e-7 (the minimum normal number of FP16 is about 6e-8, and 1e-12 underflows to 0). For FP32/BF16, the passed-in value is used directly.
- p branch rules:
  - p ∈ {0, 1, ±inf}: the identity branch is used (the reduction result is already the final value, and no root extraction is needed).
  - p ≈ 2.0: the sqrt branch is used (a high-precision sqrt instruction is used).
  - other p values: the log-exp branch is used (exp(ln(x)/p)), and large positive values may overflow in exp.
- Determinism note: deterministic implementation by default (element-wise operation, no reduction/matrix operations, naturally deterministic).
- Empty tensors and non-contiguous tensors are supported.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/arch35/test_geir_lp_norm_update_v2.cpp">test_geir_lp_norm_update_v2</a></td>
    <td>The REG_OP(LpNormUpdateV2) is added to the graph, and GE schedules the execution.</td>
  </tr>
</tbody></table>
