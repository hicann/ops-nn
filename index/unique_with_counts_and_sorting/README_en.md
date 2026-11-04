# UniqueWithCountsAndSorting

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: Flattens input tensor x and performs global deduplication, returns the unique elements y, and optionally returns the reverse indices indices from the original input to y and the occurrence counts counts of each unique element.
- Computation process:
  Let v be the flattened input, N the number of elements, and U the number of elements after deduplication. When sorted is true, y is sorted in ascending order. When the corresponding output is enabled, the following holds:

  $$y[indices[i]] = v[i], \quad 0 \le i < N$$

  $$counts[j] = \sum_{i=0}^{N-1} \mathbf{1}(indices[i] = j), \quad 0 \le j < U$$

  For example, for input x=[3, 1, 3, 2], when sorted=true, return_inverse=true, and return_counts=true, the result is y=[1, 2, 3], indices=[2, 0, 2, 1], and counts=[1, 1, 2].

## Parameters

<table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 215px">
  <col style="width: 163px">
  <col style="width: 287px">
  <col style="width: 439px">
  <col style="width: 135px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input tensor to deduplicate.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Unique elements, with the same data type as x and shape [U].</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Output</td>
      <td>When return_inverse is true, returns the index of each input element in y, starting from 0, with the same shape as x.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>counts</td>
      <td>Output</td>
      <td>When return_counts is true, returns the occurrence count of each element in y within x, with shape [U].</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>return_inverse</td>
      <td>Optional attribute</td>
      <td>Whether to return the reverse indices. Default value: false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_counts</td>
      <td>Optional attribute</td>
      <td>Whether to return the occurrence counts. Default value: false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sorted</td>
      <td>Optional attribute</td>
      <td>Whether to require the unique elements to be sorted in ascending order. Default value: true.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out_idx</td>
      <td>Optional attribute</td>
      <td>Data type of the indexes and counts. Currently only DT_INT64 is supported, and the default value is DT_INT64.</td>
      <td>TYPE</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraints

Input tensor x cannot be empty, and the number of dimensions must be greater than 0.

## Invocation Description

| Invocation method | Invocation sample | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_unique](../unique/examples/test_aclnn_unique.cpp) | Calls the UniqueWithCountsAndSorting operator through the [aclnnUnique](../unique/docs/aclnnUnique_en.md) or [aclnnUnique2](../unique/docs/aclnnUnique2_en.md) interface. |
| Graph-mode invocation | [test_geir_unique_with_counts_and_sorting](./examples/test_geir_unique_with_counts_and_sorting.cpp) | Calls the UniqueWithCountsAndSorting operator by building a graph with the [operator IR](./op_graph/unique_with_counts_and_sorting_proto.h). |
