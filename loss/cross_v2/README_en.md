# CrossV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     ×    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: computes the vector cross product of the tensors x1 and x2 along the dimension dim.
- Formula:

  $$
  out = a \times b = \left|\begin {array}{c}
  i&j&k \\
  x_1&y_1&z_1 \\
  x_2&y_2&z_2 \\
  \end{array}\right| = (y_1z_2 - y_2z_1)i -(x_1z_2-x_2z_1)j+(x_1y_2-x_2y_1)k

  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
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
      <td>a</td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8, INT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8, INT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Attribute</td>
      <td>Specifies the dimension for the computation.</td>
      <td>INT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>FLOAT16, FLOAT32, INT32, INT8, UINT8, INT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

The number of dimensions of a and b must be greater than dim, and the size of the dim dimension of a and b must be 3.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_cross_v2](./examples/test_geir_cross_v2.cpp)   | Invokes the CrossV2 operator by constructing the graph through the [operator IR](./op_graph/cross_v2_proto.h). |
