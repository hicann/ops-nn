# LpNormUpdate

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:-----|:-------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: Computes the "p-th root" step of the Lp norm. The input x is the output of LpNormReduce (that is, $\sum|x_i|^p$), and the output y is the Lp norm normalization factor. This result is usually used as the denominator for input division normalization, or directly output as the Lp norm value.
- Formula:

  $$
  y = \max\left(x^{\frac{1}{p}},\ \varepsilon_{\text{eff}}\right)
  $$

  Different computation paths are used for different values of $p$:
  - $p \in \{0, 1, +\infty, -\infty\}$: $y = \max(x,\ \varepsilon_{\text{eff}})$ (identity)
  - $p = 2$: $y = \max\left(\sqrt{x},\ \varepsilon_{\text{eff}}\right)$
  - Other finite $p$: $y = \max\left(x^{1/p},\ \varepsilon_{\text{eff}}\right)$ (including finite negative p, aligned with PyTorch)

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
|:-------|:--------------|:-----|:--------|:--------|
| x | Input | Output of LpNormReduce $\sum{\vert x_i \vert}^p$, corresponding to x in the formula. The input value is theoretically non-negative. | FLOAT16, FLOAT | ND |
| p | Attribute | <ul><li>Order of the norm, corresponding to p in the formula.</li><li>The default value is 2.</li><li>The value range is any integer (including 0 and negative values) or $\pm\infty$ ($\pm\infty$ is passed in through INT_MAX/INT_MIN mapping).</li><li>When p=2, the Sqrt fast path is used.</li><li>When p∈{0,1,±∞}, the identity path is used.</li></ul> | INT | - |
| epsilon | Attribute | <ul><li>Numerical stability constant, corresponding to $\varepsilon$ in the formula.</li><li>The default value is 1e-12.</li><li>The value range is $\geq 0$.</li><li>For FP16, $\epsilon = 0$ is not clamped (remains 0); when $0 < \epsilon \leq 10^{-7}$, it is adaptively clamped to $10^{-7}$ to prevent underflow.</li></ul> | FLOAT | - |
| y | Output | Lp norm normalization factor $\max(x^{1/p}, \varepsilon_{\text{eff}})$, corresponding to y in the formula. | FLOAT16, FLOAT | ND |

## Constraints

- Determinism statement: the implementation is deterministic by default.
- `x` is the output of LpNormReduce ($\sum|x_i|^p$), not the original input; the caller needs to execute LpNormReduceV2 first.
- `p` is an integer (INT type), and the value range is any integer (including 0 and negative values) or $\pm\infty$ ($\pm\infty$ is passed in through INT_MAX/INT_MIN mapping). p=0 and p∈{1,±∞} use the identity path; when p=2, the Sqrt fast path is used to avoid the precision loss of Power; finite negative p uses the Power path, aligned with PyTorch `torch.linalg.vector_norm`.
- The value range of `epsilon` is $\geq 0$; for FP16, `epsilon` $= 0$ is not clamped (remains 0), and when $0 < \epsilon \leq 10^{-7}$ it is adaptively clamped to $10^{-7}$ to prevent underflow.
- The rank of x and y must not exceed 8.
- The input value of `x` is theoretically non-negative (the output of LpNormReduce is $\geq 0$); for negative inputs, it is clamped to 0 before Sqrt/Power as a fallback, to avoid NaN.

## Invocation Description

| Invocation method | Invocation sample | Description |
|:--------|:--------|:-----|
| GE graph mode | [test_geir_lp_norm_update](examples/test_geir_lp_norm_update.cpp) | Calls the LpNormUpdate operator by constructing a graph through the [operator IR](op_graph/lp_norm_update_proto.h). |
