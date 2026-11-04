# GatherNd

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √     |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: For the input tensor `self` with **r≥1** dimensions and the input tensor `indices` with **q≥1** dimensions, the operator gathers data slices into the output tensor out with **(q-1) + (r - indices_shape[-1])** dimensions. indices is a **q**-dimensional integer tensor, which can be viewed as a special **q-1**-dimensional tensor composed of **index pairs** (each **index pair** is a one-dimensional tensor of length **indices_shape[-1]**, and each **index pair** points to one slice in **self**).
- Computation procedure:
  - If indices_shape[-1] > r, the scenario is invalid.
  - If indices_shape[-1] = r, the dimension of the output tensor out is q-1, that is, the shape of out is [indices_shape[0:q-1]], and the elements of out are the elements of self at the index-pair positions. (See Example 1)
  - If indices_shape[-1] < r, the dimension of the output tensor out is (q-1) + (r - indices_shape[-1]). Let c=indices_shape[-1], that is, the shape of out is [indices_shape[0:q-1],self_shape[c:r]], and `out` consists of the slices of `self` at the index-pair positions. (See Examples 2, 3, and 4)

  The restrictions on **r**, **q**, and **indices_shape[-1]** are as follows:
  - r≥1 and q≥1 must be satisfied.
  - The value of indices_shape[-1] must be between 1 (inclusive) and r (inclusive).
  - Each element of `indices` must be in the range [-s, s-1] (s is the value of each axis of self_shape), that is, -self_shape[i]≤indices[...,i]≤self_shape[i]-1.

- Examples:

  ```text
  Example 1:
    self: [[0, 1],[2, 3]]       # self_shape=[2, 2], r=2
    indices: [[0, 0], [1, 1]]   # indices_shape=[2, 2], q=2, indices_shape[-1]=2
    out: [0, 3]                 # out_shape=[2]
  Example 2:
    self: [[0, 1],[2, 3]]       # self_shape=[2, 2], r=2
    indices: [[1], [0]]         # indices_shape=[2, 1], q=2, indices_shape[-1]=1
    out: [[2, 3], [0, 1]]       # out_shape=[2, 2]
  Example 3:
    self: [[[0, 1],[2, 3]], [[4, 5],[6, 7]]]   # self_shape=[2, 2, 2], r=3
    indices: [[0, 1], [1, 0]]                  # indices_shape=[2, 2], q=2, indices_shape[-1]=2
    out: [[2, 3], [4, 5]]                      # out_shape=[2, 2]
  Example 4:
    self: [[[0, 1],[2, 3]], [[4, 5],[6, 7]]]   # self_shape=[2, 2, 2], r=3
    indices: [[[0, 1]], [[1, 0]]]              # indices_shape=[2, 1, 2], q=3, indices_shape[-1]=2
    out: [[[2, 3]], [[4, 5]]]                  # out_shape=[2, 1, 2]
  ```

## Parameters

  <table style="undefined;table-layout: fixed; width: 1479px"><colgroup>
    <col style="width: 183px">
    <col style="width: 120px">
    <col style="width: 265px">
    <col style="width: 197px">
    <col style="width: 114px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Data type</th>
        <th>Data format</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>self</td>
        <td>Input</td>
        <td>Input aclTensor.</td>
        <td>INT64, INT32, INT8, UINT8, BOOL, FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT16, UINT16, UINT32, UINT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>Input</td>
        <td>Input aclTensor.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>negativeIndexSupport</td>
        <td>Input</td>
        <td>Whether the onnx model contains negative indices.</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output aclTensor.</td>
        <td>Same as self</td>
        <td>ND</td>
      </tr>
    </tbody></table>

- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data types DOUBLE, INT16, UINT16, UINT32, and UINT64 are not supported.
- <term>Atlas training products</term> and <term>Atlas inference products</term>: the data types BFLOAT16, DOUBLE, INT16, UINT16, UINT32, and UINT64 are not supported.

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_gather_nd](examples/arch35/test_aclnn_gather_nd.cpp) | Invokes the GatherNd operator through the [aclnnGatherNd](./docs/aclnnGatherNd_en.md) interface. |
