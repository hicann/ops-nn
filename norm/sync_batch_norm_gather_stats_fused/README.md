# SyncBatchNormGatherStatsFused

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：SyncBatchNormGatherStatsFused算子用于计算在BatchNormTraining过程中的全局的均值和全局标准差倒数，以及更新全局方差。

- 计算公式：

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

  公式中的n为首轴，c为尾轴。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>表示输入数据均值，对应公式中的`mean`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>invstd</td>
      <td>输入</td>
      <td>表示输入数据标准差倒数，对应公式中的`invstd`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>runningMean</td>
      <td>输入</td>
      <td>表示计算过程中的均值，对应公式中的`runningMean`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>runningVar</td>
      <td>输入</td>
      <td>表示计算过程中的方差，对应公式中的`runningVar`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>counts</td>
      <td>输入</td>
      <td>表示输入数据元素的个数，对应公式中的`counts`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>momentum</td>
      <td>可选属性</td>
      <td>runningVar的指数平滑参数，默认值0.01。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>表示添加到方差中的值，以避免出现除以零的情况。对应公式中的`eps`</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>meanAll</td>
      <td>输出</td>
      <td>表示所有卡上数据的均值，对应公式中的`meanAll`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>invstdAll</td>
      <td>输出</td>
      <td>表示所有卡上数据的标准差的倒数，对应公式中的`invstdAll`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>runningMeanOut</td>
      <td>输出</td>
      <td>更新后的均值，对应公式中的`runningMeanOut`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>runningVarOut</td>
      <td>输出</td>
      <td>更新后的方差，对应公式中的`runningVarOut`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_BatchNormGatherStatsWithCounts](../sync_batch_norm_gather_stats_with_counts/examples/test_aclnn_BatchNormGatherStatsWithCounts.cpp) | 通过[aclnnBatchNormGatherStatsWithCounts](../sync_batch_norm_gather_stats_with_counts/docs/aclnnBatchNormGatherStatsWithCounts.md)接口方式调用SyncBatchNormGatherStatsFused算子。 |
