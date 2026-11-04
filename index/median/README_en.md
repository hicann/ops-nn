# Median

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: returns the median of input tensor x along the specified dimension dim and its index in the original input; when the number of elements is even, the smaller of the two middle values is taken.
- Computation procedure:
  Let n denote the length of dimension dim. Sort the elements along this dimension in ascending order, denote the sorted values as s, and denote the corresponding original indices as p. Then:

  $$m = \left\lfloor \frac{n - 1}{2} \right\rfloor, \quad y = s[m], \quad indices = p[m]$$

  For example, for input tensor x=[[4, 1, 3, 2], [8, 5, 7, 6]] and dim=1, the result is y=[[2], [6]] and indices=[[3], [3]].

  For floating-point inputs, if the slice for which the median is computed contains NaN, the corresponding median is NaN.

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
      <td>Input tensor for the computation.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Median. The data type is the same as x, and the shape is the same as x except that the length of the dim dimension is 1.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Output</td>
      <td>Index of the selected element on the dim dimension of the input tensor, starting from 0. The shape is the same as y.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the dimension for the computation, with the value range [-rank(x), rank(x) - 1].</li><li>The default value is -1, indicating the last dimension.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraints

The input tensor x must not be empty, and its number of dimensions must be greater than 0.

## Invocation Description

| Invocation method | Invocation sample | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_median](./examples/test_aclnn_median.cpp) | Computes the median of all elements through the [aclnnMedian](./docs/aclnnMedian_en.md) interface. |
| aclnn invocation | [test_aclnn_median_dim](./examples/test_aclnn_median_dim.cpp) | Computes the median and indices on the specified dimension through the [aclnnMedianDim](./docs/aclnnMedianDim_en.md) interface, and uses keepDim to control whether the output retains the reduced dimension. |
