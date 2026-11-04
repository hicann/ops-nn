# NanMedian

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

- Operator function: After ignoring NaN, returns the median of the input tensor x along the specified dimension dim and its index in the original input. When the number of valid elements is even, the smaller of the two middle values is taken.
- Computation flow:
  After excluding NaN along the dim dimension, sort the remaining elements in ascending order. Denote the number of valid elements as n, the sorted values as s, and the corresponding original indices as p. When n is greater than 0:

  $$m = \left\lfloor \frac{n - 1}{2} \right\rfloor, \quad y = s[m], \quad indices = p[m]$$

  For example, for the input tensor x=[[1, NaN, 3, 2], [4, 6, NaN, 5]] and dim=1, the result is y=[[2], [5]] and indices=[[3], [3]].

  If all elements in a slice are NaN, the corresponding median is NaN. For integer inputs, the computation result is the same as that of Median.

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
      <td>Median. The data type is the same as x, and the shape is the same as x except that the length of the dim dimension is 1.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Output</td>
      <td>Index of the selected element along the dim dimension of the input tensor, starting from 0. The shape is the same as y.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Optional attribute</td>
      <td><ul><li>Dimension along which the computation is performed, with a value range of [-rank(x), rank(x) - 1].</li><li>Default value: -1, which indicates the last dimension.</li></ul></td>
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
| aclnn invocation | [test_aclnn_nan_median](./examples/test_aclnn_nan_median.cpp) | Compute the median of all elements after ignoring NaN through the [aclnnNanMedian](./docs/aclnnNanMedian_en.md) API. |
| aclnn invocation | [test_aclnn_nan_median_dim](./examples/test_aclnn_nan_median_dim.cpp) | Compute the median and indices along the specified dimension after ignoring NaN through the [aclnnNanMedianDim](./docs/aclnnNanMedianDim_en.md) API, and use keepDim to control whether the output retains the reduced dimension. |
