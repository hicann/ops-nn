# SwiGluGrad

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

- Description: Performs the backward computation of [aclnnSwiGlu](../swi_glu/docs/aclnnSwiGlu_en.md), and completes the SwiGlu backward gradient computation of x.

- Formula:

  <p style="text-align: center">
  out<sub>A</sub> = yGrad<sub>i</sub>*[sigmoid(A)+sigmoid(A)*(1-sigmoid(A))*A]*B
  </p>
  <p style="text-align: center">
  out<sub>B</sub> = yGrad<sub>i</sub>*sigmoid(A)*A
  </p>
  A denotes the first half of the tensor after the input x is split into two halves along the specified dim dimension, and B denotes the second half of the tensor after x is evenly split. out<sub>A</sub> and out<sub>B</sub> are merged into out.

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
      <td>yGrad</td>
      <td>Input</td>
      <td>Input yGrad<sub>i</sub> in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Inputs A and B in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Attribute</td>
      <td><ul><li>Index of the dimension to be split.</li><li>Value range: [-x.dim(), x.dim()-1].</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Merge of out<sub>A</sub> and out<sub>B</sub> in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>

  </tbody></table>

- <term>Atlas inference products</term>: the supported data types are FLOAT16 and FLOAT32.

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_swi_glu_grad](./examples/test_aclnn_swi_glu_grad.cpp) | Invokes the SwiGluGrad operator through the [aclnnSwiGluGrad](./docs/aclnnSwiGluGrad_en.md) interface.    |
| Graph-mode invocation | -   | Invokes the SwiGluGrad operator through graph construction with the [operator IR](./op_graph/swi_glu_grad_proto.h). |
