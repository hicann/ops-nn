# TensorScatterSub

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

> Note: Ascend 950PR/950DT is directly supported by the kernel in this directory. Atlas A2/A3 products are supported through the graph-mode fusion rule (TensorScatterSubFusionPass, fused into TensorMove + ScatterNdSub), without an independent kernel.

## Function

- Operator function: copies x to the output y, then subtracts the corresponding slice in updates from y one by one at the positions specified by indices. Repeated indexes are subtracted serially in input order, and the result is deterministic.

- Formula:

$$
y = copy(x)\\
for\ i \in [0, N):\ y[indices_i] \mathrel{-}= updates_i
$$

Here, N is the product of indices.shape[:-1], that is, the number of slices.

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
      <td>The source tensor for the tensor_scatter_sub computation, the x in the formula.</td>
      <td>FLOAT16, FLOAT, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>The index tensor. The length K of the last dimension indicates that each index covers the first K dimensions of x.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>The slice data to be subtracted, the updates in the formula.</td>
      <td>FLOAT16, FLOAT, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The result tensor of the computation. The shape is the same as that of x.</td>
      <td>FLOAT16, FLOAT, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The dtypes of x, updates, and y must be consistent. The dtype of indices is INT32 or INT64.
- The shape of y is exactly the same as the shape of x.
- The dimension range of x is 1D to 8D.
- Let K=indices.shape[-1]. K must be less than or equal to the number of dimensions of x. The shape of updates must be indices.shape[:-1] concatenated with x.shape[K:].
- The value of the j-th index component in indices must be within the valid range [0,x.shape[j]) of the j-th dimension of x. Out-of-bounds indexes are invalid inputs, and the kernel skips these entries.
- When indices contain repeated indexes, subtraction is performed serially one by one in input order, and the result is deterministic.
- x, indices, and updates can be empty tensors (with 0 elements). In this case, y is a copy of x.

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
    <td><a href="./examples/test_geir_tensor_scatter_sub.cpp">test_geir_tensor_scatter_sub</a></td>
    <td>Invokes the TensorScatterSub operator through graph construction using the <a href="./op_graph/tensor_scatter_sub_proto.h">operator IR</a>.</td>
  </tr>
</tbody>
</table>
