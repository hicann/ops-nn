# SyncBatchNormGatherStats

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: the SyncBatchNormGatherStats operator collects the means and variances of all devices, and updates the global mean and variance.

- Computation formula:

  $$
  batchMean = \frac{\sum^N_{i=0}{totalSum[i]}}{\sum^N_{i=0}{sampleCount[i]}}
  $$

  $$
  batchVar = \frac{\sum^N_{i=0}{totalSquareSum[i]}}{\sum^N_{i=0}{sampleCount[i]}} - batchMean^2
  $$

  $$
  batchInvstd = \frac{1}{\sqrt{batchVar + ε}}
  $$

  $$
  runningMean = runningMean*(1-momentum) + momentum*batchMean
  $$

  $$
  runningVar = runningVar*(1-momentum) + momentum*(batchVar*   \frac{\sum^N_{i=0}
  {sampleCount[i]}}{\sum^N_{i=0}{sampleCount[i]}-1})
  $$

## Parameters

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
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
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
<tr>
    <td>total_sum</td>
    <td>Input</td>
    <td>Denotes the channel feature sums of the devices, corresponding to totalSum in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
</tr>
<tr>
    <td>total_square_sum</td>
    <td>Input</td>
    <td>Denotes the channel feature squares of the devices, corresponding to totalSquareSum in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
</tr>
<tr>
    <td>sample_count</td>
    <td>Input</td>
    <td>Denotes the sample counts of the devices, corresponding to sampleCount in the formula.</td>
    <td>INT32</td>
    <td>ND</td>
</tr>
<tr>
    <td>mean</td>
    <td>Input</td>
    <td>Denotes the mean during the computation, corresponding to runningMean in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
</tr>
<tr>
    <td>variance</td>
    <td>Input</td>
    <td>Denotes the variance during the computation, corresponding to runningVar in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
</tr>
<tr>
    <td>momentum</td>
    <td>Attribute</td>
    <td>Exponential smoothing parameter of runningMean and runningVar.</td>
    <td>FLOAT</td>
    <td>-</td>
</tr>
<tr>
    <td>eps</td>
    <td>Attribute</td>
    <td>Offset used to prevent division by 0.</td>
    <td>FLOAT</td>
    <td>-</td>
</tr>
<tr>
    <td>batch_mean</td>
    <td>Output</td>
    <td>Denotes the global batch mean, corresponding to batchMean in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
</tr>
<tr>
    <td>batch_invstd</td>
    <td>Output</td>
    <td>Denotes the reciprocal of the standard deviation, corresponding to batchInvstd in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
</tr>
<tr>
    <td>mean</td>
    <td>Output</td>
    <td>Denotes the updated mean, corresponding to runningMean in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
</tr>
<tr>
    <td>variance</td>
    <td>Output</td>
    <td>Denotes the updated variance, corresponding to runningVar in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
</tr>
</tbody></table>

- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: the data type BFLOAT16 is not supported for the parameters `totalSum`, `totalSquareSum`, `mean`, `variance`, `batchMean`, and `batchInvstd`.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_sync_batch_norm_gather_stats](../sync_batch_norm_gather_stats/examples/test_aclnn_sync_batch_norm_gather_stats.cpp) | Calls the SyncBatchNormGatherStats operator through the [aclnnSyncBatchNormGatherStats](../sync_batch_norm_gather_stats/docs/aclnnSyncBatchNormGatherStats_en.md) interface. |
| Graph mode | [test_geir_sync_batch_norm_gather_stats](../sync_batch_norm_gather_stats/examples/test_geir_sync_batch_norm_gather_stats.cpp) | Calls the SyncBatchNormGatherStats operator by constructing a graph through the [operator IR](op_graph/sync_batch_norm_gather_stats_proto.h).         |
