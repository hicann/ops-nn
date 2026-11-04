# L2Normalize

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: performs L2 normalization on the input tensor `x` along the axis list specified by `axis`, and outputs `y` with the same shape and dtype as `x`.
- Formula:

  $$
  s = \sum_{i \in axis} x_i^2 \quad (keepdims)
  $$

  $$
  y = x / \sqrt{\max(s, eps)}
  $$

  Here `eps` is the lower bound of the sum of squares, with max applied before sqrt; when `eps > 0`, division by zero for all-zero or extremely small vectors is avoided.
  The public contract does not restrict the sign of `eps`; when `eps <= 0`, computation follows the formula above directly. When `eps` is a non-finite value (`inf`/`nan`), the behavior is undefined, and the caller should pass a finite value.

## Parameters

<table style="table-layout: fixed; width: 1005px"><colgroup>
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
      <td>Input feature tensor to be normalized, that is, `x` in the formula.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>L2 normalization result tensor, that is, `y` in the formula, with the same shape and data type as `x`.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Optional attribute</td>
      <td>Normalization axis list, that is, `axis` in the formula. It supports joint reduction over multiple axes and negative index conversion, with the default value `{}`; when omitted or an empty list, no cross-element reduction happens, and computation is element-wise.</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Optional attribute</td>
      <td>Lower bound of the sum of squares, that is, `eps` in the formula, with the default value 1e-4.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

### Product Differences

The public GEIR interfaces of all supported products stay consistent: `x` and `y` support FLOAT16, FLOAT32, and the ND format, `y` has the same shape and data type as `x`, and the actual rank at runtime is 1-8; the semantics and default values of `axis` and `eps` are also consistent. The specific capabilities and contract boundaries of each product are as follows.

| Product | Implementation and invocation path | Static shape capability | Dynamic shape capability | Shape/rank and extended scenarios |
| :--- | :--- | :--- | :--- | :--- |
| <term>Ascend 950PR&950DT products</term> | Capability is provided by the <term>Ascend 950PR&950DT products</term> implementation of this repository, invoked through GE graph mode. | FLOAT16 and FLOAT32, ND→ND; supports 1-8 dim concrete shapes, with the output having the same shape and dtype as the input. | ND→ND; graph deduction supports unknown dimensions (`-1`) and unknown rank (`{-2}`), which must be resolved to a 1-8 dim concrete shape before execution. | 0-dim scalars are not supported; empty tensors with a dimension of 0 and non-contiguous tensors are supported. |
| <term>Atlas A3 products</term><br><term>Atlas A2 products</term><br><term>Atlas 200I/500 A2 inference products</term><br><term>Atlas inference products</term><br><term>Atlas training products</term> | Capability is provided by the existing CANN implementation, invoked through GE graph mode. | FLOAT16 and FLOAT32, ND→ND; supports 1-8 dim concrete shapes, with the output having the same shape and dtype as the input. | ND→ND; the existing CANN registration supports dynamic shapes, and the actual rank at runtime still needs to satisfy the 1-8 dim public contract. | 0-dim scalars are not supported; the existing public contract does not explicitly promise empty tensors, unknown rank, or non-contiguous tensors, and this document does not treat these scenarios as external capabilities of this type of product. |

## Constraints

- `x` and `y` support only FLOAT16 and FLOAT32 (with identical data types), the data format supports only ND, the number of dimensions is 1-8, `y.shape = x.shape`, and 0-dim scalar inputs are not supported.
- When `axis` is omitted or an empty list, no reduction happens; in this case, `y = x / sqrt(max(x * x, eps))` is computed element-wise. When the list is not empty, the value range of each element is `[-rank(x), rank(x))`, negative index conversion is supported, and duplicates converted to the same dimension are treated as one reduction axis.
- On the <term>Ascend 950PR&950DT products</term>, empty tensors (when a dimension is 0, the output is an empty tensor with the same dimensions), dynamic shapes (a dimension is `-1`), unknown dimension counts (the dims of the shape are `{-2}`, with the actual rank deduced during graph compilation), and non-contiguous tensors are supported.
- On the <term>Ascend 950PR&950DT products</term>, this operator is not an in-place implementation, and the input `x` remains unchanged after execution; the caller must ensure that the underlying memory of `y` and `x` does not overlap.
- On the <term>Ascend 950PR&950DT products</term>, FLOAT16 inputs are upcast to FLOAT32 for computation inside the operator and then converted back to FLOAT16 for output.

## Invocation Description

| Invocation method | Invocation sample | Description | Supported products |
|---------|----|------|---------|
| GE graph mode | [test_geir_l2_normalize](examples/test_geir_l2_normalize.cpp) | Calls the L2Normalize operator by constructing a graph through the [operator IR](op_graph/l2_normalize_proto.h). | All supported products |
