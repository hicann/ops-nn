# GroupNormSiluQuant

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Description: computes the group normalization of input self, and outputs the mean meanOut, the reciprocal of the standard deviation rstdOut, and the result out of quantizing the Silu output.

- Formula:
  - **GroupNorm:**

    Let $E[x] = \bar{x}$ denote the mean of $x$ and $Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$ denote the variance of $x$. Then

    $$
    \left\{
    \begin{array} {rcl}
    groupNormOut& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
    meanOut& &= E[x]\\
    rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
    \end{array}
    \right.
    $$

  - **Silu:**

    $$
    siluOut = \frac{groupNormOut}{1+e^{-groupNormOut}}
    $$

  - **Quant:**

    $$
    out = round(siluOut / quantScale)
    $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>Denotes the input tensor to be normalized, corresponding to `x` in the formula. It has 2 to 8 dimensions, with dimension 0 as N and dimension 1 as C.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Denotes the scaling tensor after normalization, corresponding to $\gamma$ in the formula. The data type is consistent with `x`, and the shape is `[C]` (C is dimension 1 of `x`).</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Denotes the offset tensor after normalization, corresponding to $\beta$ in the formula. The data type is consistent with `x`, and the shape is `[C]` (C is dimension 1 of `x`).</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quantScale</td>
      <td>Input</td>
      <td>Denotes the quantization scale factor, corresponding to `quantScale` in the formula. The shape is `[1]` (per-tensor) or `[C]` (per-channel, where C is dimension 1 of `x`).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_groups</td>
      <td>Attribute</td>
      <td>Denotes dividing dimension 1 (C) of `x` into `num_groups` groups; C must be divisible by num_groups.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td>Denotes the perturbation added to the variance during normalization, corresponding to $\epsilon$ in the formula, with the default value 1e-05.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_silu</td>
      <td>Attribute</td>
      <td>Indicates whether to apply Silu activation to the normalization result; the default value is true. When set to false, the activation is skipped and quantization is performed directly.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>Output</td>
      <td>Denotes the quantized output tensor, corresponding to `yOut` in the formula. The shape is consistent with `x`.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>meanOut</td>
      <td>Output</td>
      <td>Denotes the mean of each group, corresponding to `meanOut` in the formula. The data type is consistent with `x`, and the shape is (N, num_groups).</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstdOut</td>
      <td>Output</td>
      <td>Denotes the reciprocal of the standard deviation of each group, corresponding to `rstdOut` in the formula. The data type is consistent with `x`, and the shape is (N, num_groups).</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- `x` has 2 to 8 dimensions, with dimension 0 as N and dimension 1 as C; C must be divisible by `num_groups`.
- The shapes of `gamma` and `beta` must be `[C]`, and their data types must be consistent with `x`.
- The shape of `quantScale` must be `[1]` (per-tensor) or `[C]` (per-channel).
- This section describes the constraints of the operator (graph mode). The aclnn interface has its own constraints; for details, see [aclnnGroupNormSiluQuant](docs/aclnnGroupNormSiluQuant_en.md).
- Empty tensors are supported: N (dimension 0) and C (dimension 1) must be greater than 0, and the other dimensions can be 0; in this case, `yOut` is empty, `meanOut` is filled with 0, and `rstdOut` is filled with NAN.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_group_norm_silu_quant](examples/arch22/test_aclnn_group_norm_silu_quant.cpp) | Invokes the GroupNormSiluQuant operator through the [aclnnGroupNormSiluQuant](docs/aclnnGroupNormSiluQuant_en.md) interface. |
| Graph-mode invocation | [test_geir_group_norm_silu_quant](examples/arch35/test_geir_group_norm_silu_quant.cpp) | Invokes the GroupNormSiluQuant operator through graph construction with the [operator IR](op_graph/group_norm_silu_quant_proto.h). |
