# RepeatInterleaveGrad

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

- Operator function: The backward pass of the RepeatInterleave operator. It performs ReduceSum on the axis dimension of the y_grad tensor according to repeats.

## Parameters

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 60px">
  <col style="width: 60px">
  <col style="width: 310px">
  <col style="width: 150px">
  <col style="width: 60px">
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
      <td>y_grad</td>
      <td>Input</td>
      <td>Indicates the input tensor to be reduced by ReduceSum.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>repeats</td>
      <td>Input</td>
      <td>Indicates the number of repetitions.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x_grad</td>
      <td>Output</td>
      <td>Indicates the output tensor after ReduceSum is complete.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates the dimension on which ReduceSum is performed. The value range is [-n, n), where n is the number of dimensions of y_grad.</li><li>The default value is -1.</li></ul></td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_repeat_interleave_grad](./examples/test_aclnn_repeat_interleave_grad.cpp) | Calls the RepeatInterleaveGrad operator through the [aclnnRepeatInterleaveGrad](./docs/aclnnRepeatInterleaveGrad_en.md) interface. |
| Graph-mode invocation | [test_geir_repeat_interleave_grad](./examples/test_geir_repeat_interleave_grad.cpp) | Calls the RepeatInterleaveGrad operator by building a graph through the [operator IR](./op_graph/repeat_interleave_grad_proto.h). |
