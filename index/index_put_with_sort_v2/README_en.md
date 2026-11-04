# index_put_with_sort_v2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×      |

## Function

- Operator function: performs the scatter operation, iterates over the corresponding values according to the values of pos_idx, and accumulates/replaces them at the positions of self pointed to by linear_index.

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
      <td>self</td>
      <td>Input/Output</td>
      <td>Destination tensor of the accumulate/replace operation</td>
      <td>INT64, INT32, FLOAT, FLOAT16, BFLOAT16, DT_BF16, INT8, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>linear_index</td>
      <td>Input</td>
      <td>Index tensor, pointing to the corresponding positions of self</td>
      <td>INT64, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pos_idx</td>
      <td>Input</td>
      <td>Index tensor, pointing to the corresponding positions of values</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>Input</td>
      <td>Source tensor of the accumulate/replace operation</td>
      <td>INT64, INT32, FLOAT, FLOAT16, BFLOAT16, DT_BF16, INT8, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Sample code                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [IndexPutWithSortV2](../index_put_v2/op_api/aclnn_index_put_impl.cpp) | Invokes the index_put_with_sort_v2 operator through the [DeterministicProcess](../index_put_v2/docs/aclnnIndexPutImpl_en.md) interface. |
