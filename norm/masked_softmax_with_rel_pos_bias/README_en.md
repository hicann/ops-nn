# MaskedSoftmaxWithRelPosBias

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    ×     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: Replaces the part that uses window attention to compute softmax in swinTransformer.
- Formula:

  $$
  out=\operatorname{softmax}(scaleValue*x+attenMaskOptional+relativePosBias)
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 312px">
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
      <td>Corresponds to x in the formula. The shapes 4D (B*W, N, S1, S2) and 5D (B, W, N, S1, S2) are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Optional input</td>
      <td>Corresponds to attenMaskOptional in the formula. It is a null pointer when not needed. The shapes 3D (W, S1, S2), 4D (W, 1, S1, S2), and 5D (1, W, 1, S1, S2) are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>relativePosBias</td>
      <td>Input</td>
      <td>Corresponds to relativePosBias in the formula. The shapes 3D (N, S1, S2), 4D (1, N, S1, S2), and 5D (1, 1, N, S1, S2) are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Optional attribute</td>
      <td>Corresponds to scaleValue in the formula.</td>
      <td>DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>innerPrecisionMode</td>
      <td>Optional attribute</td>
      <td>Precision mode parameter.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Corresponds to out in the formula. The shapes 4D (B*W, N, S1, S2) and 5D (B, W, N, S1, S2) are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas training products</term>: BFLOAT16 is not supported.
- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: The data types of all inputs and outputs do not support BFLOAT16.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_masked_softmax_with_rel_pos_bias](examples/test_aclnn_masked_softmax_with_rel_pos_bias.cpp) | Calls the MaskedSoftmaxWithRelPosBias operator through the [aclnnMaskedSoftmaxWithRelPosBias](docs/aclnnMaskedSoftmaxWithRelPosBias_en.md) interface. |
| Graph mode | [test_aclnn_masked_softmax_with_rel_pos_bias](examples/test_aclnn_masked_softmax_with_rel_pos_bias.cpp)  | Calls the MaskedSoftmaxWithRelPosBias operator by building a graph through [operator IR](op_graph/masked_softmax_with_rel_pos_bias_proto.h).         |
