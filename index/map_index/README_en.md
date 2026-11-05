# MapIndex

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: computes the mapped index according to the hierarchical index relationship of the input tensors. The output is a 0-dim scalar.
- Computing principle: computes the target index value through the hierarchical mapping relationship among `x` (input index), `data_seq` (data sequence), and the optional `level_index` (hierarchical index).

## Parameters

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>The input index, a 1-dim tensor.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>data_seq</td>
    <td>Input</td>
    <td>The data sequence, a 1-dim tensor whose length must be an integer multiple of the length of x.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>level_index</td>
    <td>Optional input</td>
    <td>The hierarchical index, a 1-dim tensor whose length must be equal to data_seq_length / x_length.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>transpose</td>
    <td>Optional attribute</td>
    <td><ul><li>Reserved attribute, only false is supported.</li><li>Passing true will be rejected.</li><li>The default value is false.</li></ul></td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>The mapped index result, a 0-dim scalar tensor.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- Length range of `x`: [1, 24000].
- Upper limit of `data_seq_length / x_length`: 256.
- `x` and `data_seq` do not support dynamic shapes (unknown dim -1 / unknown rank -2 will be rejected).
- When `level_index` is an unknown dim (-1) or an unknown rank (-2), the length check is skipped.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation example</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph mode invocation</td>
    <td><a href="./examples/test_geir_map_index.cpp">test_geir_map_index</a></td>
    <td>This operator supports only graph mode (GE) invocation and does not provide the two-segment aclnn API.</td>
  </tr>
</tbody></table>
