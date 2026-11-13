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

- Operator function: it keeps the C axis of the input and reduces all other axes, outputting the per-channel element sum and square sum.
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
    <td>The input tensor to be reduced, corresponding to x in the formulas.</td>
    <td>FLOAT16, BFLOAT16, FLOAT</td>
    <td>NCHW, NHWC, NCDHW</td>
  </tr>
  <tr>
    <td>sum</td>
    <td>Output</td>
    <td>The per-channel element sum over all axes except C, corresponding to sum in the formulas.</td>
    <td>FLOAT</td>
    <td>Same as x</td>
  </tr>
  <tr>
    <td>square_sum</td>
    <td>Output</td>
    <td>The per-channel square sum over all axes except C, corresponding to square_sum in the formulas.</td>
    <td>FLOAT</td>
    <td>Same as x</td>
  </tr>
</tbody></table>

## Constraints

- <term>Ascend 950PR&950DT products</term>:
  - x supports NCHW tensors with 2 to 4 dimensions, NHWC tensors with 4 dimensions, and NCDHW tensors with 5 dimensions.
  - sum and square_sum both have a 1D shape [C], where C is the channel dimension of x. Their data formats are the same as x.
  - Graph mode supports dynamic shapes and dynamic ranks.
  - Empty tensors are supported. When C = 0, the outputs are empty tensors. When C > 0 and the reduction set is empty, both outputs are all-zero tensors of shape [C].

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
    <td>Calls the two-phase <a href="./docs/aclnnBatchNormReduce_en.md">aclnnBatchNormReduce API</a>. Refer to that API document for applicable products and parameter constraints.</td>
  </tr>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/test_geir_bn_training_reduce.cpp">test_geir_bn_training_reduce</a></td>
    <td>Builds and executes the graph through the <a href="./op_graph/bn_training_reduce_proto.h">BNTrainingReduce operator IR</a>.</td>
  </tr>
</tbody></table>
