# Where

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                         |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function

Operator function: finds the positions of the non-zero or True elements in tensor `x`. Let the number of dimensions of tensor `x` be D and the number of non-zero elements be N; then the shape of the returned `y` is D * N, and each column represents the position coordinates of one non-zero element.

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>Input tensor.</td>
      <td>INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64<br>
          QINT8, QUINT8, QINT32, FLOAT16, FLOAT, DOUBLE, BOOL<br>
          COMPLEX64, COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Position tensor of the non-zero elements or True values in the input tensor x.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraints

- None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_where](./examples/test_geir_where.cpp)   | Invokes the where operator by building a graph through the [operator IR](./op_graph/where_proto.h). |
