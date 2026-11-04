# LinearIndex

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    ×   |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: Indexing method that maps the element positions of a multi-dimensional array or matrix to a single integer.

## Operator Specifications

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
      <td>indices</td>
      <td>Input</td>
      <td>Input indexes.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shape</td>
      <td>Input</td>
      <td>Shape information of the input data.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Output</td>
      <td>Index to output.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Optional attribute</td>
      <td>Axis to compute. Default value: -1.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>combine</td>
      <td>Optional attribute</td>
      <td>Whether to combine axes. Default value: false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

None
