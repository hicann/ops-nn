# SyncBNTrainingUpdate

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

- Operator function: the SyncBNTrainingUpdate operator is used to update the global mean during BatchNormTraining.

- Formula:

$$
running_mean_update = (mean * momentum) + running_mean * (1 - momentum)
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
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Represents the mean of the input data, corresponding to `mean` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>running_mean</td>
      <td>Input</td>
      <td>Represents the mean in the computation process, corresponding to `running_mean` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>momentum</td>
      <td>Attribute</td>
      <td>Exponential smoothing parameter of runningMean, with a default value of 0.1.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>running_mean_update</td>
      <td>Output</td>
      <td>Updated mean, corresponding to `running_mean_update` in the formula.</td>
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
| aclnn interface  | [test_aclnn_BatchNormGatherStatsWithCounts](../sync_batch_norm_gather_stats_with_counts/examples/test_aclnn_BatchNormGatherStatsWithCounts.cpp) | Invokes the SyncBNTrainingUpdate operator through the [aclnnBatchNormGatherStatsWithCounts](../sync_batch_norm_gather_stats_with_counts/docs/aclnnBatchNormGatherStatsWithCounts_en.md) interface. |
| Graph mode | [test_geir_sync_bn_training_update](../sync_bn_training_update/examples/test_geir_sync_bn_training_update.cpp) | Invokes the SyncBNTrainingUpdate operator by building a graph through the [operator IR](op_graph/sync_bn_training_update_proto.h).         |
