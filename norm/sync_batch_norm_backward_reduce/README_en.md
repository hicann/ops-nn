# SyncBatchNormBackwardReduce

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: SyncBatchNormBackwardReduce computes the weight gradient gradWeight and the intermediate quantity sumDyXmu required by the BatchNorm operation during backpropagation.

- Formula:

$$
sumDyXmu = {sumDyDxPad} - {sumDy} * {mean}
$$

$$
gradWeight = ({sumDyDxPad} - {sumDy} * {mean}) * invertStd
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
      <th>Input/Output</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>sum_dy</td>
      <td>Input</td>
      <td>Accumulated sum of the forward output gradients, corresponding to `sumDy` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_dy_dx_pad</td>
      <td>Input</td>
      <td>Corresponding to `sumDyDxPad` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Mean of the input data, corresponding to `mean` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>invert_std</td>
      <td>Input</td>
      <td>Reciprocal of the standard deviation of the input data, corresponding to `invertStd` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_dy_xmu</td>
      <td>Output</td>
      <td>Sum of the products of the forward output gradients and the centered input data, corresponding to `sumDyXmu` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Gradient of the scaling parameter, corresponding to `gradWeight` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_batch_norm_reduce_backward](../sync_batch_norm_backward_reduce/examples/test_aclnn_batch_norm_reduce_backward.cpp) | Invoke the SyncBatchNormBackwardReduce operator through the [aclnnBatchNormReduceBackward](../sync_batch_norm_backward_reduce/docs/aclnnBatchNormReduceBackward_en.md) API. |
| Graph mode | [test_geir_sync_batch_norm_backward_reduce](../sync_batch_norm_backward_reduce/examples/test_geir_sync_batch_norm_backward_reduce.cpp) | Invoke the SyncBatchNormBackwardReduce operator by building a graph through the [operator IR](op_graph/sync_batch_norm_backward_reduce_proto.h).         |
