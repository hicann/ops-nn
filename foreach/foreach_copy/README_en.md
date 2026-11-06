# ForeachCopy

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: Implements the copying of content between two tensor lists.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = x_i (i=0,1,...n-1)
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>x</td>
      <td>Input</td>
      <td>Represents the input tensor list for content copying, corresponding to `x` in the formula. The data types of all tensors in this parameter are the same.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16, INT8, UINT8, INT16, UINT16, UINT32, UINT64, INT64, DOUBLE, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Represents the output tensor list for content copying, corresponding to `y` in the formula.</li><li>The data format is consistent with the data format of input parameter `x`, and the shape size is greater than or equal to the shape size of input parameter `x`.</li><li>The data types of all tensors in this parameter are the same.</li><li>The data type has a certain correspondence with the data type of input parameter `x`: <ul><li>When the data type of input parameter `x` is FLOAT32, the output data type supports FLOAT16 or BFLOAT16.</li><li>When the data type of input parameter `x` is FLOAT16, the output data type supports FLOAT32.</li><li>When the data type of input `x` is not FLOAT32 or FLOAT16, the output data type must be consistent with the data type of input `x`.</li></ul></li></ul></td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16, INT8, UINT8, INT16, UINT16, UINT32, UINT64, INT64, DOUBLE, BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- Non-contiguous tensors are not supported for the output.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_foreach_copy](examples/test_aclnn_foreach_copy.cpp) | Calls the ForeachCopy operator through the [aclnnForeachCopy](docs/aclnnForeachCopy_en.md) interface. |
| Graph-mode invocation | -  | Calls the ForeachCopy operator by building a graph through the [operator IR](op_graph/foreach_copy_proto.h).         |
