# Renorm

English | [简体中文](./README.md)

## Applicable Products

| Product                                                        | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √    |
| <term>Atlas A3 products</term>     |    ×    |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: returns a tensor in which each sub-tensor of the input tensor self along dimension dim is normalized, so that the p-norm of the sub-tensor is lower than the maxnorm value.

- Computation formula:

  $$
  output_i=\left\{
  \begin{aligned}
  input_i,\quad ||input_i||_p ≤ maxnorm \\
  \frac {input_i} {max(||input_i||_p, eps)} \cdot maxnorm,\quad ||input_i||_p > maxnorm
  \end{aligned}
  \right.
  $$

  where:
  $i$ is a tensor slice along a dimension determined by dim:

  - When p > 0:

    $$
    ||input_i||_p =\left\{
    \begin{aligned}
    \sqrt[p]{reduce\_sum(|input_i|^p, dim)},\quad (p > 0) \& (p \ ≠ inf) \\
    max(|input_i|),\quad p = inf
    \end{aligned}
    \right.
    $$

  - When p == 0, $||input_i||$ counts the non-0 values in input:

    $$
    ||input_i||={\sum_{i=1}^n {\mathbb{I}}}(input_i ≠ 0)
    $$

    where ${\mathbb{I}} = 1$ when $input_i ≠ 0$, and ${\mathbb{I}} = 0$ when $input_i = 0$.

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
      <td>Denotes the input of the renormalization computation, corresponding to `input` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>Attribute</td>
      <td>Denotes the norm, corresponding to `p` in the formula. The value is greater than or equal to 0.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Attribute</td>
      <td>Denotes the specified dimension direction for computing the norm. It corresponds to `i` in the formula. The value range is [-the number of dimensions of x, the number of dimensions of x - 1].</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxnorm</td>
      <td>Attribute</td>
      <td>Denotes the maximum allowed normalization value. It corresponds to `maxnorm` in the formula. The value is greater than or equal to 0. If the `p`-norm of the corresponding dimension (determined by the `p` value) during the computation is greater than `maxnorm`, the values of that dimension are normalized with respect to the `p`-norm and multiplied by `maxnorm`. If the `p`-norm of the corresponding dimension (determined by the `p` value) during the computation is smaller than `maxnorm`, the tensor of that dimension is output unchanged.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the final output, corresponding to `output` in the formula. Its data type and shape are consistent with input parameter `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_renorm](examples/test_aclnn_renorm.cpp) | Calls the Renorm operator through the [aclnnRenorm](docs/aclnnRenorm&aclnnInplaceRenorm_en.md) interface. |
| aclnn interface  | [test_aclnn_inplace_renorm](examples/test_aclnn_inplace_renorm.cpp) | Calls the Renorm operator through the [aclnnInplaceRenorm](docs/aclnnRenorm&aclnnInplaceRenorm_en.md) interface. |
| aclnn interface  | [test_aclnn_embedding_renorm](examples/test_aclnn_embedding_renorm.cpp) | Calls the Renorm operator through the [aclnnEmbeddingRenorm](docs/aclnnEmbeddingRenorm_en.md) interface. |
