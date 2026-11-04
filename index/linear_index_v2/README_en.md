# LinearIndexV2

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √   |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |

## Function

- Operator function: An indexing method that maps the element positions of multidimensional arrays / matrices to a single integer.

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
      <td>stride</td>
      <td>Input</td>
      <td>The stride.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices_list</td>
      <td>Input</td>
      <td>The index list.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value_size</td>
      <td>Input</td>
      <td>The size of the value.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Output</td>
      <td>The index to be output.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None
