# KthValue

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: Returns the k-th minimum value of the input tensor x on the specified dimension dim and its index in the original input.
- Computation process:
  For each one-dimensional slice taken along the dim dimension, let the values sorted in ascending order be s, and let p[j] denote the position of s[j] before the slice is sorted (counting from 0). Then:

  $$y = s[k - 1], \quad indices = p[k - 1]$$

  For example, for the input tensor x=[[3, 1, 2], [6, 4, 5]], k=2, and dim=1, the result is y=[[2], [5]] and indices=[[2], [2]]. Here, the indexes of 2 and 5 in their original rows are both 2.

## Parameters

<table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 215px">
  <col style="width: 163px">
  <col style="width: 287px">
  <col style="width: 439px">
  <col style="width: 135px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input tensor to be computed.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The k-th minimum value. Its data type is the same as x, and its shape is the same as x except that the length of the dim dimension is 1.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Output</td>
      <td>Index of the selected element on the dim dimension of the input tensor, starting from 0. Its shape is the same as y.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>Required attribute</td>
      <td>Specifies the k-th minimum value, with the value range [1, x.shape[dim]].</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the dimension for computation, with the value range [-rank(x), rank(x) - 1].</li><li>The default value is -1, indicating the last dimension.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraints

The input tensor x cannot be empty, and the number of dimensions must be greater than 0.

## Invocation Description

| Invocation method | Invocation sample | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_kthvalue](./examples/test_aclnn_kthvalue.cpp) | Computes the k-th minimum value and index on the specified dimension through the [aclnnKthvalue](./docs/aclnnKthvalue_en.md) interface. |
