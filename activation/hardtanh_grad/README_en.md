# HardtanhGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Backward of the Hardtanh activation function.
- Formula:

  $$
  grad\_self_{i} =
  \begin{cases}
    0,\ \ \ \ \ \ \ if \ \ self_{i}>=max \\
    0,\ \ \ \ \ \ \  if\ \ self_{i}<=min \\
    1,\ \ \ \ \ \ \ \ \ \ \ \ otherwise \\
  \end{cases}
  $$

  $$
  res_{i} = grad\_output_{i} \times grad\_self_{i}
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 350px">
  <col style="width: 200px">
  <col style="width: 150px">
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
      <td>result</td>
      <td>Input</td>
      <td>Gradient output by the previous step in backpropagation, which is grad_output in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Forward input data, which is self in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>min_val</td>
      <td>Attribute</td>
      <td>Lower limit of the linear range, which is min in the formula.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
      <tr>
      <td>max_val</td>
      <td>Attribute</td>
      <td>Upper limit of the linear range, which is max in the formula.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Computed gradient, which is res in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

result, grad, and out must have the same shape and data type.

## Invocation Description

| Invocation method | Invocation sample | Description |
| ---- |---- | ---- |
| aclnn invocation | [test_aclnn_hardtanh_grad](examples/test_aclnn_hardtanh_grad.cpp) | Invokes the HardtanhGrad operator through [aclnnHardtanhBackward](docs/aclnnHardtanhBackward_en.md).|
| Graph-mode invocation | - | Invokes the HardtanhGrad operator through graph construction with the [operator IR](op_graph/hardtanh_grad_proto.h). |
