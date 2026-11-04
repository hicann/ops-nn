# NonZeroWithValue

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: finds the non-zero elements of the 2D input `x`. With a static max-size, the three outputs `value`, `index`, and `count` are returned at one time, and the valid length is given by `count`.
- Computation description:
  - `value`: the values of the non-zero elements.
  - `index`: the coordinates of the non-zero elements, flattened in coordinate-major order (`[2, numel]`, the first half is row numbers and the second half is column numbers).
  - `count`: the number of non-zero elements.
  - The non-zero determination follows the IEEE semantics: nan and ±inf are counted as non-zero, and -0.0 is regarded as zero. The traversal is in row-major order, and the result is deterministic.

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
      <td>The target tensor to be computed, strictly 2D.</td>
      <td>DOUBLE, FLOAT, FLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Output</td>
      <td>The values of the non-zero elements. The data type is the same as x. It is a static max-size output, and the shape is [row*col].</td>
      <td>DOUBLE, FLOAT, FLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Output</td>
      <td>The coordinates of the non-zero elements, flattened in coordinate-major order. The shape is [2*row*col].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>count</td>
      <td>Output</td>
      <td>The number of non-zero elements. The shape is [1].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Attribute</td>
      <td>Identifies the layout of index. Only true (coordinate-major order) is supported.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td>Controls the output data type of index. Only INT32 is supported.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The input `x` is strictly 2D, and 12 data types are supported: DOUBLE, FLOAT, FLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, and BOOL. The data type of `value` is the same as `x`, and `index`/`count` are always INT32.
- The total number of elements of the input `x` (row×col) must not exceed INT32_MAX (2³¹-1). The coordinate output `index` is INT32, and when the total number of elements exceeds the limit, the coordinates cannot be represented, so the tiling check rejects it.
- The outputs are static max-size, and the valid length is given by `count`.
- Deterministic Computation: deterministic by default.

## Invocation Description

| Invocation method   | Sample code                                                     | Description                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph-mode invocation | [test_geir_non_zero_with_value](examples/test_geir_non_zero_with_value.cpp) | Invoke the NonZeroWithValue operator by constructing a graph through the [operator IR](op_graph/non_zero_with_value_proto.h). |
