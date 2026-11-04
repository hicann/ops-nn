# UnsortedSegmentProd

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Description: Computes the segmented product of a tensor
- Formula description:
    $$
    y[i] = {\prod}_{j...:segment_ids[j...]==i} x[j...]
    $$
    Here, the product is computed over all positions j... that satisfy segment_ids[j...] == i. If a segment i has no corresponding elements, then y[i] = 1.

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
      <td>Input data</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32, INT64, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>segment_ids</td>
      <td>Input</td>
      <td>Segment indices</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_segments</td>
      <td>Input</td>
      <td>Number of segments</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output value information</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32, INT64, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_unsorted_segment_prod](examples/test_geir_unsorted_segment_prod.cpp) | Invokes the UnsortedSegmentProd operator by building a graph through the [operator IR](op_graph/unsorted_segment_prod_proto.h). |
