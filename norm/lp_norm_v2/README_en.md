# LpNormV2

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function

- Description: returns the matrix norm or vector norm of the given tensor.

- Formula: supports the computation of the 0/1/2 norm, the infinite norm, and other norms in which `p` is of the float type.
  - 0-norm:

    $$
    \Vert x \Vert = \sum_{i=1}^{N}{(x_i ≠ 0)}
    $$

  - 1-norm:

    $$
    \Vert x \Vert = max(\sum_{i=1}^{N}{\vert x_i \vert}, epsilon)
    $$

  - 2-norm (default value):

    $$
    \Vert x \Vert_2 = max((\sum_{i=1}^{N}{\vert x_i \vert^2})^{\frac{1}{2}}, epsilon)
    $$

  - Infinite norm:

    $$
    \Vert x \Vert_\infty = max(\max\limits_{i}{\vert x_i \vert}, epsilon)
    $$

    $$
    \Vert x \Vert_{-\infty} = max(\min\limits_{i}{\vert x_i \vert}, epsilon)
    $$

  - p-norm:

    $$
    \Vert x \Vert_p = max((\sum_{i=1}^{N}{\vert x_i \vert^p})^{\frac{1}{p}}, epsilon)
    $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 312px">
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
      <td>x in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>Optional attribute</td>
      <td>Represents the type of the norm, which is p in the formula.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axes</td>
      <td>Optional attribute</td>
      <td>Dimensions on which the norm of self is computed.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepdim</td>
      <td>Optional attribute</td>
      <td>Determines whether the output tensor keeps the axes specified by the dim parameter.</td>
      <td>BOOL </td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td>Prevents the norm result from being 0 to ensure numerical stability. Default value: 1e-12f.</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation sample

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface | [test_aclnn_norm](./examples/test_aclnn_norm.cpp) | Invokes the LpNormV2 operator through the [aclnnNorm](docs/aclnnNorm_en.md) interface. |
| aclnn interface | [test_aclnn_linalg_vector_norm](./examples/test_aclnn_LinalgVectorNorm.cpp) | Invokes the LpNormV2 operator through the [aclnnLinalgVectorNorm](docs/aclnnLinalgVectorNorm_en.md) interface. |
| Graph mode | - | Invokes the LpNormV2 operator by building a graph through the [operator IR](op_graph/lp_norm_v2_proto.h). |
