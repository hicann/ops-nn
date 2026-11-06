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
len = output_offsets[i+1] - output_offsets[i]
$$

$$
output_permute[output_offsets[i]:output_offsets[i+1]] = arange(input_offsets[permute[i]],input_offsets[permute[i]]+len)
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
      <td>input_offsets</td>
      <td>Input</td>
      <td>Denotes the exclusive offsets of the table-level lengths.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_offsets</td>
      <td>Input</td>
      <td>Denotes the exclusive offsets of the table-level permutation lengths.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_permute</td>
      <td>Output</td>
      <td>Denotes the output in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
        <tr>
      <td>output_size</td>
      <td>Attribute</td>
      <td><ul><li>Length of the output result.</li></ul></td>
      <td>INT32, INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

Let P be the length of permute. The lengths of input_offsets and output_offsets must both be P+1.
When P is 0, output_size must be 0. When P is greater than 0, output_size must be greater than or equal to P.

input_offsets and output_offsets must start at 0 and, when P is greater than 0, must be strictly increasing.
The last value of input_offsets equals the total number of input-side elements.
The last value of output_offsets must equal output_size.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_expand_into_jagged_permute.cpp](examples/test_aclnn_expand_into_jagged_permute.cpp) | Invokes the ExpandIntoJaggedPermute operator through the [aclnnExpandIntoJaggedPermute](docs/aclnnExpandIntoJaggedPermute_en.md) API. |
