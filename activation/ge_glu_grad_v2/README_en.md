# GeGluGradV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     ×    |

## Function

Description: Performs the backward computation of [aclnnGeGlu](../ge_glu_v2/docs/aclnnGeGlu_en.md) and [aclnnGeGluV3](../ge_glu_v2/docs/aclnnGeGluV3_en.md).

## Parameters

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>gradOutput</td>
      <td>Input</td>
      <td>Input gradOutput in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Input self in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gelu</td>
      <td>Input</td>
      <td>Input gelu in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Attribute</td>
      <td>The value range is [-self.dim(), self.dim()-1].</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>approximate</td>
      <td>Attribute</td>
      <td><ul><li>The value range is 0('none') and 1('tanh').</li><li>Atlas inference products support only 1('tanh').</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
      <tr>
      <td>activateLeft</td>
      <td>Attribute</td>
      <td><ul><li>Direction of the data block on which the activation function operates.</li><li>false means performing activate on the right side.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>Output</td>
      <td>gradInput in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- The aclnnGeGluBackward API does not include the activateLeft parameter.
- <term>Atlas inference products</term> and <term>Atlas training products</term>: The data types of gradOutput, self, gelu, and gradInput support FLOAT32 and FLOAT16.

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_ge_glu_backward](./examples/test_aclnn_ge_glu_backward.cpp) | Calls the GeGluGradV2 operator through the [aclnnGeGluBackward](./docs/aclnnGeGluBackward_en.md) API.    |
| aclnn invocation | [test_aclnn_ge_glu_v3_backward](./examples/test_aclnn_ge_glu_v3_backward.cpp) | Calls the GeGluGradV2 operator through the [aclnnGeGluV3Backward](./docs/aclnnGeGluV3Backward_en.md) API.    |
| Graph-mode invocation | -   | Calls the GeGluGradV2 operator by constructing a graph through the [operator IR](./op_graph/ge_glu_grad_v2_proto.h). |
