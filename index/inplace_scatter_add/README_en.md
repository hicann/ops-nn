# InplaceScatterAdd

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

- Operator function: Adds the values in updates to the first dimension of the input tensor var according to the given indices.

- Formula:

  $$
  var[indices]  +=  updates
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 40px">
  <col style="width: 60px">
  <col style="width: 260px">
  <col style="width: 60px">
  <col style="width: 40px">
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
      <td>var</td>
      <td>Input/Output</td>
      <td>Tensor to be updated, which is var in the formula.</td>
      <td>FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Indexes of the first dimension of var. The shape supports only 1D, and the value range of the data is [0, var.dim[0]).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Values to be added to var.</td>
      <td>FLOAT, INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------- | ---------------- | -------------------------------------------- |
| aclnn invocation  | [test_aclnn_index_add.cpp](examples/test_aclnn_index_add.cpp) | Invokes the InplaceScatterAdd operator through the [aclnnIndexAdd](docs/aclnnIndexAdd_en.md) interface. |
