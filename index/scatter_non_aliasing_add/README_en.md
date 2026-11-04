# ScatterNonAliasingAdd

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

- Operator function: ScatterNonAliasingAdd performs a "copy first, then scatter-accumulate" operation. The operator copies the entire input tensor x to output y, and then accumulates the values in updates to the corresponding positions of y according to the multi-dimensional index positions specified by indices. Output y does not share memory with input x (Non-Aliasing).

- Formula:

$$
y = \text{copy}(x)
$$
$$
y[\text{indices}[i]] \mathrel{+}= \text{updates}[i], \quad i = 0, 1, \ldots, N-1
$$

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>Source tensor. The output is its copy with updates scattered and accumulated.</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Multi-dimensional index tensor, with rank(indices) ≥ 2 and shape[-1] = K ≤ rank(x). It specifies the scatter target positions.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Tensor of update values to be accumulated, with shape = indices.shape[:-1] + x.shape[K:].</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, with shape = x.shape and dtype = x.dtype.</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Output y does not share memory with input x (Non-Aliasing), and y is an independently allocated tensor.
- All inputs and outputs use FORMAT_ND (contiguous storage), and non-contiguous tensors are not supported.
- The rank of indices must be greater than or equal to 2, and the last dimension indicates the index depth K.
- Repeated indices are accumulated multiple times in the original order of indices, and the result is deterministic.
- Each index value in indices must satisfy 0 ≤ index < x.shape[dim]. Negative indices and out-of-bounds indices are not supported. The operator does not validate index validity. You must ensure that the indices are valid; otherwise, the result is undefined.
- Empty tensors (shapes containing 0 dimensions) are supported. In this case, a copy of x is returned, and scatter add is not performed.

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | - | Invoke the ScatterNonAliasingAdd operator through the [operator IR](op_graph/scatter_non_aliasing_add_proto.h) API. |
