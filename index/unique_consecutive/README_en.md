# UniqueConsecutive

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √  |
| <term>Atlas A3 products</term>    |    ×    |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |

## Function

- Operator function: removes the duplicate elements after each element. When dim is not empty, removes the duplicate tensors after each tensor along the corresponding dimension.

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
      <td>Input parameter for the UniqueConsecutive deduplication computation.</td>
      <td>BF16, FLOAT16, FLOAT, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>return_idx</td>
      <td>Input attribute</td>
      <td>Whether to return the positions of the elements in self within the output y.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>return_counts</td>
      <td>Input attribute</td>
      <td>Whether to return the repetition counts of the elements in the output y.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Input attribute</td>
      <td>Deduplication dimension.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out_idx</td>
      <td>Input attribute</td>
      <td>Specifies the types of the output idx and count, with the default value INT64.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output after removing consecutive duplicates of elements.</td>
      <td>BF16, FLOAT16, FLOAT, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>idx</td>
      <td>Output</td>
      <td>Returns the positions of the elements in self within the output y, effective when return_idx is True.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>count</td>
      <td>Output</td>
      <td>Returns the repetition counts of the elements in the output y, effective when return_counts is True.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_unique_consecutive](./examples/test_aclnn_unique_consecutive.cpp) | Calls the UniqueConsecutive operator through the [aclnnUniqueConsecutive](./docs/aclnnUniqueConsecutive_en.md) interface. |
