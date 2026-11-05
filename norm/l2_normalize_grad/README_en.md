# L2NormalizeGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Backward operator of the forward L2Normalize ($y = x/max(sqrt(sum(x^2, dim)), eps)$). The three inputs are `x` (forward input), `y` (forward output, that is, the normalized x), and `dy` (upstream gradient), and the single output is `dx` (gradient with respect to x).

- Formula (reduction is performed along the reduction axis set specified by the attribute `dim`, and keepdims broadcasts back to the full shape):

  $$
  n = \max\left(\sqrt{\sum_{dim} x^2}, \ eps\right)
  $$

  $$
  s = \sum_{dim} (y \cdot dy)
  $$

  $$
  dx = \frac{dy - y \cdot s}{n}
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
      <td>Denotes the input of the forward operator, corresponding to <code>x</code> in the formula. 1 to 8 dimensions are supported.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Input</td>
      <td>Denotes the output of the forward operator, that is, the normalized <code>x</code>, corresponding to <code>y</code> in the formula. Its shape and data type are consistent with the input parameter <code>x</code>.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>Denotes the gradient passed back in the backward pass, corresponding to <code>dy</code> in the formula. Its shape and data type are consistent with the input parameter <code>x</code>.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Attribute</td>
      <td>Denotes the set of normalization axes. The default value is an empty array <code>[]</code>. The value range of each element is [-<code>x</code>.dim(), <code>x</code>.dim()-1]. A negative axis number is automatically converted to a positive index (<code>dim + rank(x)</code>). Elements can be duplicated, out of order, or mixed positive and negative, elements can be duplicated, out of order, or mixed positive and negative. After deduplication and sorting, the reduction axes must form a contiguous interval and be continuously distributed dimension by dimension (such as <code>[1]</code>, <code>[1,2]</code>). Axis sets with non-adjacent axis numbers (such as <code>[0,2]</code>) are not supported. The array length does not exceed 20.</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td>Denotes the lower bound of the denominator. The default value is 1e-4.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>Output</td>
      <td>Denotes the gradient with respect to <code>x</code>, corresponding to <code>dx</code> in the formula. Its shape and data type are consistent with the input parameter <code>x</code>.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- When `x` is an empty tensor, `dx` is also an empty tensor.

- The shapes and data types of the input parameters `x`, `y`, and `dy` must be exactly the same, and the output `dx` is the same as them.

- The input parameter `x` supports 1 to 8 dimensions. The reduction axis length and the tensor element count have no additional upper bound (when the capacity of a single UB is exceeded, the operator internally processes the data in blocks along the reduction axis).

- The reduction axis set of the attribute `dim` (after converting negative values and deduplication) must form a contiguous interval, such as `[1]`, `[1,2]`, and `[0,1,2]`. Axis sets with non-adjacent axis numbers, such as `[0,2]` and `[1,3]`, are not supported.

- The array length of the attribute `dim` does not exceed 20. An error is reported when an element value is out of [-`x`.dim(), `x`.dim()-1].

- When the attribute `dim` is not passed or is passed as an empty array, no reduction is performed. In this case, `dx` is computed by the elementwise formula: $dx = (dy - y \cdot y \cdot dy) / \max(|x|, eps)$.

## Invocation Description

This operator is a backward operator inside the GE gradient graph and **does not provide an aclnn single-operator interface** (there is no `aclnnL2NormalizeGrad`). It is invoked only through a GE graph.

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_l2_normalize_grad](examples/test_geir_l2_normalize_grad.cpp) | Calls the L2NormalizeGrad operator by building a graph through [operator IR](op_graph/l2_normalize_grad_proto.h). |
