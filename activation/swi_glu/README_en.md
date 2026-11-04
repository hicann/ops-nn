# SwiGlu

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
|  <term>kirin X90 processor series products</term>    |     √    |
|  <term>Kirin 9030 processor series products</term>    |     √    |

## Function

- Description: the Swish gated linear unit activation function, which implements the SwiGlu computation of x.

- Formula:

  <p style="text-align: center">
  out<sub>i</sub> = SwiGlu(x<sub>i</sub>)=Swish(A<sub>i</sub>)*B<sub>i</sub>
  </p>
  Here, A<sub>i</sub> is the first half of the tensor after x<sub>i</sub> is split in half along the specified dim dimension, and B<sub>i</sub> is the second half of the tensor after x<sub>i</sub> is split in half along the specified dim dimension.

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
      <td>x</td>
      <td>Input</td>
      <td>The input x in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Attribute</td>
      <td><ul><li>The dimension index to split. The corresponding axis of x is split in half.</li><li>The value range is [-x.dim(), x.dim()-1].</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The out<sub>i</sub> in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- For <term>Atlas inference products</term>, the supported data types are FLOAT16 and FLOAT32.
- For <term>kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>, BFLOAT16 is not supported.

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_swi_glu](./examples/test_aclnn_swi_glu.cpp) | Invokes the SwiGlu operator through the [aclnnSwiGlu](./docs/aclnnSwiGlu_en.md) API.    |
| Graph-mode invocation | -   | Invokes the SwiGlu operator through graph construction using the [operator IR](./op_graph/swi_glu_proto.h). |
