# IndexPutImpl

English | [简体中文](./README.md)


## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √    |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |

## Function

- Description: according to the indices, the operator replaces or accumulates the data at the corresponding coordinates of the input self with the input values.
- Formula:

  - accumulate = False:

    $$
    self[indices] = values
    $$

  - accumulate = True:

    $$
    self[indices]  = self[indices]  + values
    $$

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
        <td>selfRef</td>
        <td>Input</td>
        <td>self in the formula. The data type is consistent with values.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>Input</td>
        <td>indices in the formula.</td>
        <td>INT32, INT64, BOOL</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>values</td>
        <td>Input</td>
        <td>values in the formula.</td>
        <td>Consistent with selfRef</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>accumulate</td>
        <td>Input</td>
        <td>Flag of the operation type, accumulation or update, a boolean value on the Host side.<ul><li>When accumulate is True, the operation is accumulation;</li><li>when accumulate is False, the operation is update.</li></ul></td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>unsafe</td>
        <td>Input</td>
        <td>Flag of whether to skip the out-of-range index check.<ul><li>When unsafe is True, the indices are not checked for out-of-range values;</li><li>When unsafe is False, the indices are checked for out-of-range values.</li></ul></td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  - <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>: the BFLOAT16 data type is not supported.

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_index_put_v2](examples/arch35/test_aclnn_index_put_v2.cpp) | Calls the IndexPutImpl operator through the [aclnnIndexPutImpl](./docs/aclnnIndexPutImpl_en.md) interface. |
