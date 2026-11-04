# Index

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: performs multi-dimensional advanced indexing on input `x` by the indices specified in `indices`, takes the data at the corresponding coordinates, and obtains output `y`. The relationships between each input and the function are as follows:
  - `indexed_sizes`: a one-dimensional INT64 tensor whose length equals the number of dimensions of `x`; it marks dimension by dimension whether each dimension of `x` participates in indexing: the value `1` indicates that the dimension is selected by the index (consuming one `indices` input), and a value greater than `1` indicates that the dimension is kept as a whole.
  - `indexed_strides`: a one-dimensional INT64 tensor whose length equals the number of dimensions of `x`; it gives the stride on each dimension of `x` when fetching data by index.
  - `indices`: dynamic inputs, whose number equals the number of dimensions with the value `1` in `indexed_sizes`, serving as the indices of the corresponding indexed dimensions; multiple `indices` are aligned according to the broadcast rules.
  - `y`: the data fetched from `x` according to the preceding rules, with the data type consistent with `x`.

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
      <td>Input data.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT64, INT8, UINT8, BOOL, COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indexed_sizes</td>
      <td>Input</td>
      <td>Number of indices.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indexed_strides</td>
      <td>Input</td>
      <td>Index strides.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Indices.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Data fetched according to the indices.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT64, INT8, UINT8, BOOL, COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas training products</term>, <term>Atlas inference products</term>: the data types BFLOAT16 and COMPLEX64 are not supported.
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: COMPLEX64 is not supported.

## Constraints

- <term>Ascend 950PR&950DT products</term>: the shapes of the tensors in indices must be the same or satisfy a broadcast relationship.
- Other chips: the shapes of the tensors in indices must be exactly the same.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_index](./examples/test_aclnn_index.cpp) | Invokes the Index operator through the [aclnnIndex](./docs/aclnnIndex_en.md) interface. |
