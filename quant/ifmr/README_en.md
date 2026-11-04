# IFMR

English | [简体中文](./README.md)


## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

A quantization method based on reconstruction of the input feature map.

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 150px">
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
      <td>data</td>
      <td>Input</td>
      <td>Tensor of the feature map.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>data_min</td>
      <td>Input</td>
      <td>Tensor of the minimum value of the feature map.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>data_max</td>
      <td>Input</td>
      <td>Tensor of the maximum value of the feature map.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cumsum</td>
      <td>Input</td>
      <td>Tensor of the cumulative sum of the data.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min_percentile</td>
      <td>Attribute</td>
      <td>Minimum initialization percentile.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_percentile</td>
      <td>Attribute</td>
      <td>Maximum initialization percentile.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>search_range</td>
      <td>Attribute</td>
      <td>Search range.</td>
      <td>LISTFLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>search_step</td>
      <td>Attribute</td>
      <td>Search step.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>with_offset</td>
      <td>Attribute</td>
      <td>Whether to use the offset.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_bits</td>
      <td>Attribute</td>
      <td>Quantization bit width, an optional attribute, with the default value 8.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Output</td>
      <td>Optimal scale.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Output</td>
      <td>Optimal offset.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_ifmr](./examples/test_geir_ifmr.cpp)  | Calls the IFMR operator by building a graph through [operator IR](./op_graph/ifmr_proto.h).         |
