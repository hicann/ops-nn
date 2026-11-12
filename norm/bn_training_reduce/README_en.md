# BNTrainingReduce

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: it keeps the C axis of the input and reduces all other axes, outputting the per-channel element sum and square sum. <term>Ascend 950PR&950DT products</term> support
  NCHW with 2 to 4 dimensions, NHWC with 4 dimensions, and NCDHW with 5 dimensions.
- Formula:

  $$
  sum_c = \sum_{n,h,w}x_{n,c,h,w}
  $$

  $$
  square\_sum_c = \sum_{n,h,w}x_{n,c,h,w}^{2}
  $$

The formulas use 4D NCHW as an example. All other supported ranks and data formats also keep the C axis and reduce all other axes; the NCDHW scenario additionally includes the D axis.

## Parameters

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 520px">
<col style="width: 300px">
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
    <td>The training activations to be counted.</td>
    <td>FLOAT16, BFLOAT16, FLOAT</td>
    <td>NCHW, NHWC, NCDHW</td>
  </tr>
  <tr>
    <td>sum</td>
    <td>Output</td>
    <td>The per-channel element sum obtained by reducing along the N, H, and W axes.</td>
    <td>FLOAT</td>
    <td>Same as x</td>
  </tr>
  <tr>
    <td>square_sum</td>
    <td>Output</td>
    <td>The per-channel square sum obtained by reducing along the N, H, and W axes.</td>
    <td>FLOAT</td>
    <td>Same as x</td>
  </tr>
</tbody></table>

## Constraints

- <term>Ascend 950PR&950DT products</term>:
  - NCHW with 2 to 4 dimensions, NHWC with 4 dimensions, and NCDHW with 5 dimensions are supported; NDC1HWC0 is not supported. The output format is the same as x.
  - Graph mode supports dynamic shapes (-1 for an unknown dimension) and dynamic ranks (-2 for an unknown rank).
- The logical shapes of sum and square_sum are both 1D [C], with a length equal to the C dimension of x, and the data type is fixed to FLOAT.
- FLOAT16 and BFLOAT16 inputs perform squaring and accumulation at FLOAT precision.
- The operator has no attributes and supports empty tensors. When C = 0, the outputs are empty tensors. When C > 0 and the reduction set is empty, both outputs are all-zero tensors of shape [C].

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn invocation</td>
    <td><a href="./examples/test_aclnn_batch_norm_reduce.cpp">test_aclnn_batch_norm_reduce</a></td>
    <td>Calls the BNTrainingReduce operator through the two-phase aclnnBatchNormReduce API.</td>
  </tr>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/test_geir_bn_training_reduce.cpp">test_geir_bn_training_reduce</a></td>
    <td>Builds the graph through the operator IR and calls the BNTrainingReduce operator.</td>
  </tr>
</tbody></table>
