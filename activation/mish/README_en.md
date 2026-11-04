# Mish

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: A self-regularized non-monotonic neural network activation function.

- Formula:

$$out_i = self_i \times \tanh(\mathrm{Softplus}(\mathrm{self}_i))$$

where $\mathrm{Softplus}(self_i) = \ln{(1 + \exp{(self_i)})}$

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
      <td>self</td>
      <td>Input</td>
      <td>Input parameter for the Mish activation computation, corresponding to self in the calculation formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter of the Mish activation computation, corresponding to out in the calculation formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

 - <term>Atlas training products</term>: The supported data types are FLOAT and FLOAT16.

## Constraints

- Input shape restriction: The input dimensions cannot exceed 8, and only 0 to 8 dimensions are supported.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_mish](./examples/test_aclnn_mish.cpp) | Calls the Mish operator through the [aclnnMish&aclnnInplaceMish](./docs/aclnnMish&aclnnInplaceMish_en.md) interface. |
| aclnn invocation | [test_aclnn_inplace_mish](./examples/test_aclnn_inplace_mish.cpp) | Calls the Mish operator through the [aclnnMish&aclnnInplaceMish](./docs/aclnnMish&aclnnInplaceMish_en.md) interface. |
| Graph mode | - | Calls the Mish operator by building a graph through [operator IR](./op_graph/mish_proto.h). |
