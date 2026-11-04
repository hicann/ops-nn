# ExpandIntoJaggedPermute

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √   |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: expands the permutation indices of sparse data from the table dimension to the batch dimension, which applies to the case where sparse features have different batch sizes on different ranks.
- Formula:

$$
len = outputOffset[i+1] - outputOffset[i]
$$

$$
outputPermuteOut[outputOffset[i]:outputOffset[i+1]] = arange(inputOffset[permute[i]],inputOffset[permute[i]]+len)
$$

## Parameters

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 250px">
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
      <td>permute</td>
      <td>Input</td>
      <td>Denotes the table-level permutation indices.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputOffset</td>
      <td>Input</td>
      <td>Denotes the exclusive offsets of the table-level lengths.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>outputOffset</td>
      <td>Input</td>
      <td>Denotes the exclusive offsets of the table-level permutation lengths.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>outputPermute</td>
      <td>Output</td>
      <td>Denotes the output in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
        <tr>
      <td>outputSize</td>
      <td>Attribute</td>
      <td><ul><li>Length of the output result.</li></ul></td>
      <td>INT32, INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

The length of inputOffset and outputOffset is 1 more than that of permute.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_expand_into_jagged_permute.cpp](tests/ut/op_kernel/test_expand_into_jagged_permute.cpp) | Invokes the ExpandIntoJaggedPermute operator through the [aclnnExpandIntoJaggedPermute](docs/aclnnExpandIntoJaggedPermute_en.md) API. |
