# SyncBatchNormBackwardElemt

English | [简体中文](./README.md)


## Applicable Products

| Product | Support |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: the SyncBatchNormBackwardElemt operator computes the elementwise gradients of the input tensor, so that the model parameters can be updated during backpropagation.

- Formula:

  $$
  gradInput = ((gradOut - meanDy) - (input - mean) * (invstd^2 * meanDyXmu)) * invstd * weight
  $$

## Parameters

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 145px">
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
      <td>grad_output</td>
      <td>Input</td>
      <td>Denotes the differential of the forward output, corresponding to `gradOut` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>save_input</td>
      <td>Input</td>
      <td>Denotes the input for the BatchNorm computation, corresponding to `input` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Denotes the mean of the input data, corresponding to `mean` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>invstd</td>
      <td>Input</td>
      <td>Denotes the reciprocal of the standard deviation of the input data, corresponding to `invstd` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Denotes the weight tensor, corresponding to `weight` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean_dy</td>
      <td>Input</td>
      <td>Denotes the mean of the sample mean sums of the output gradients, corresponding to `meanDy` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean_dy_xmu</td>
      <td>Input</td>
      <td>Denotes the mean of the products of the sample mean sums and the input gradients, corresponding to `meanDyXmu` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>Output</td>
      <td>Denotes the gradient of the input tensor, corresponding to `gradInput` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

The supported combinations of the parameters grad_output, save_input, mean, invstd, weight, mean_dy, mean_dy_xmu, and grad_input are as follows:

- <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, and <term>Atlas A2 products</term>:

  | grad_output | save_input | mean | invstd | weight | mean_dy | mean_dy_xmu | grad_input |
  | --------| --------| --------| --------| --------| --------|  --------|:------ |
  | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |
  | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 |
  | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |
  | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT16 |

- <term>Atlas training products</term>:

  | grad_output | save_input | mean | invstd | weight | mean_dy | mean_dy_xmu | grad_input |
  | --------| --------| --------| --------| --------| --------|  --------|:------ |
  | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |
  | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_batch_norm_elemt_backward](../sync_batch_norm_backward_elemt/examples/test_aclnn_batch_norm_elemt_backward.cpp) | Calls the SyncBatchNormBackwardElemt operator through the [aclnnBatchNormElemtBackward](../sync_batch_norm_backward_elemt/docs/aclnnBatchNormElemtBackward_en.md) interface. |
| Graph mode | [test_geir_sync_batch_norm_backward_elemt](../sync_batch_norm_backward_elemt/examples/test_geir_sync_batch_norm_backward_elemt.cpp) | Calls the SyncBatchNormBackwardElemt operator by building a graph through [operator IR](op_graph/sync_batch_norm_backward_elemt_proto.h).         |
