# DataFormatDimMap

English | [简体中文](./README.md)


## Applicable Products

| Product | Support |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: according to the dimension mapping between the source data format and the target data format, the operator converts the input dimension indices to the corresponding dimension indices under the target format.
- Computation example: the input dimension indices are `[0, 1, 2, 3]`, the source format is `NHWC`, and the target format is `NCHW`; the output is `[0, 2, 3, 1]`. That is, dimension 1 (H) of NHWC corresponds to dimension 2 of NCHW, dimension 2 (W) corresponds to dimension 3, and dimension 3 (C) corresponds to dimension 1.

## Parameters

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>Dimension index tensor, with values in [-N, N), where N is the dimension count of src_format (N=4 for a 4-dimensional format and N=5 for a 5-dimensional format, that is, the value range of x is [-4, 4) or [-5, 5)). Negative indices are supported, where -i denotes the i-th dimension counted from the end (for example, -1 denotes the last dimension).</td>
    <td>INT32, INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>src_format</td>
    <td>Optional attribute</td>
    <td><ul><li>Source data format, with the enumeration values: "NCHW", "NHWC", "NCDHW", "NDHWC".</li><li>The default value is "NHWC".</li></ul></td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>dst_format</td>
    <td>Optional attribute</td>
    <td><ul><li>Target data format, with the enumeration values: "NCHW", "NHWC", "NCDHW", "NDHWC".</li><li>The default value is "NCHW".</li></ul></td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Mapped dimension index tensor, with the same shape as the input x.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- src_format and dst_format support only the enumeration values: "NCHW", "NHWC" (4-dimensional), "NCDHW", and "NDHWC" (5-dimensional).
- src_format and dst_format must contain the same character set (that is, the same dimension letters, in possibly different orders), that is, both must be 4-dimensional formats or both 5-dimensional formats.
- The values of the elements in the input x are in [-N, N), where N is the dimension count of src_format (N=4 for a 4-dimensional format and N=5 for a 5-dimensional format); inputs beyond this range are not guaranteed to obtain correct mapping results.
- The shape of the input x has no more than 8 dimensions.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph mode | [test_geir_data_format_dim_map](./examples/test_geir_data_format_dim_map.cpp) | Calls the DataFormatDimMap operator by building a graph through [operator IR](./op_graph/data_format_dim_map_proto.h). |
