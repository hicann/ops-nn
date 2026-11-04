# SyncBatchNormGatherStatsFused

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: the SyncBatchNormGatherStatsFused operator computes the global mean and the reciprocal of the global standard deviation during BatchNormTraining, and updates the global variance.

- Formula:

  $$
  countsSum = \sum_{n=0}^{N-1} counts[n]
  $$

  $$
  var[n,c] = \frac{1}{invstd[n,c]^2} - eps
  $$

  $$
  meanDiff[n,c] = mean[n,c] - meanAll[c]
  $$

  $$
  varSum[c] = \sum_{n=0}^{N-1} (var[n,c] + meanDiff[n,c]^2) \times counts[n]
  $$

  $$
  unbiasedVar[c] = \frac{varSum[c]}{countsSum - 1}
  $$

  $$
  meanAll[c] = \frac{\sum_{n=0}^{N-1} mean[n,c] \times counts[n]}{countsSum}
  $$

  $$
  invstdAll[c] = \frac{1}{\sqrt{\frac{varSum[c]}{countsSum} + eps}}
  $$

  $$
  runningMeanOut[c] = (1 - momentum) \times runningMean[c] + momentum \times meanAll[c]
  $$

  $$
  runningVarOut[c] = (1 - momentum) \times runningVar[c] + momentum \times unbiasedVar[c]
  $$

  In the formula, n is the leading axis and c is the trailing axis.

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
      <td>runningMean</td>
      <td>Input</td>
      <td>Denotes the mean during the computation, corresponding to `runningMean` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>runningVar</td>
      <td>Input</td>
      <td>Denotes the variance during the computation, corresponding to `runningVar` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>counts</td>
      <td>Input</td>
      <td>Denotes the number of elements of the input data, corresponding to `counts` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>momentum</td>
      <td>Optional attribute</td>
      <td>Exponential smoothing parameter of runningVar. The default value is 0.01.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td>Denotes the value added to the variance to avoid division by zero. It corresponds to `eps` in the formula</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>meanAll</td>
      <td>Output</td>
      <td>Denotes the mean of the data on all cards, corresponding to `meanAll` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>invstdAll</td>
      <td>Output</td>
      <td>Denotes the reciprocal of the standard deviation of the data on all cards, corresponding to `invstdAll` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>runningMeanOut</td>
      <td>Output</td>
      <td>Updated mean, corresponding to `runningMeanOut` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>runningVarOut</td>
      <td>Output</td>
      <td>Updated variance, corresponding to `runningVarOut` in the formula.</td>
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
| aclnn invocation  | [test_aclnn_BatchNormGatherStatsWithCounts](../sync_batch_norm_gather_stats_with_counts/examples/test_aclnn_BatchNormGatherStatsWithCounts.cpp) | Invokes the SyncBatchNormGatherStatsFused operator through the [aclnnBatchNormGatherStatsWithCounts](../sync_batch_norm_gather_stats_with_counts/docs/aclnnBatchNormGatherStatsWithCounts_en.md) API. |
