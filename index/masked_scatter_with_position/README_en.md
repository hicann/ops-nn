# MaskedScatterWithPosition

English | [简体中文](./README.md)


## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √  |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |

## Function

- Operator function: according to the mask tensor, the operator performs simulated broadcasting to the shape of x. At the positions where the mask is True, the operator copies the corresponding values from updates to the corresponding positions of the input x. Here position is the prefix-sum tensor of the mask before broadcasting.

- Example:
  - x=[[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]
  - masked=[[0, 0, 0, 1, 1]]
  - position=[[0, 0, 0, 1, 2]]
  - updates=[[0, 1, 2, 3, 4] ,[5, 6, 7, 8, 9]]
  - The result of maskedscatterwithposition(x, masked, position, updates) is [[0, 0, 0, 0, 1],[0, 0, 0, 2, 3]].

## Parameters

<table style="undefined;table-layout: fixed; width: 920px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 290px">
  <col style="width: 260px">
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
      <td>Input, the original data.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, UINT8, INT8, INT16, INT32, INT64, DOUBLE, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Input</td>
      <td>Input mask.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>position</td>
      <td>Input</td>
      <td>Prefix sum of the mask.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Data to be filled in.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, UINT8, INT8, INT16, INT32, INT64, DOUBLE, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td>Output result.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, UINT8, INT8, INT16, INT32, INT64, DOUBLE, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_masked_scatter_with_position.cpp](./examples/test_aclnn_masked_scatter_with_position.cpp) | Calls the L0 interface of the MaskedScatterWithPosition operator through the [aclnnInplaceMaskedScatter](../masked_scatter/op_api/aclnn_masked_scatter.cpp) interface. |
