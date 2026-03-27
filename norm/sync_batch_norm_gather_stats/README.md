# SyncBatchNormGatherStats

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：SyncBatchNormGatherStats算子用于收集所有device的均值和方差，更新全局的均值和方差。


- 计算公式：

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

## 参数说明

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
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
<tr>
    <td>total_sum</td>
    <td>输入</td>
    <td>表示各设备的通道特征和，对应公式中的totalSum。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
</tr>
<tr>
    <td>total_square_sum</td>
    <td>输入</td>
    <td>表示各设备的通道特征平方，对应公式中的totalSquareSum。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
</tr>
<tr>
    <td>sample_count</td>
    <td>输入</td>
    <td>表示各设备的样本计数，对应公式中的sampleCount。</td>
    <td>INT32</td>
    <td>ND</td>
</tr>
<tr>
    <td>mean</td>
    <td>输入</td>
    <td>表示计算过程中的均值，对应公式中的runningMean。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
</tr>
<tr>
    <td>variance</td>
    <td>输入</td>
    <td>表示计算过程中的方差，对应公式中的runningVar。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
</tr>
<tr>
    <td>momentum</td>
    <td>属性</td>
    <td>runningMean和runningVar的指数平滑参数。</td>
    <td>FLOAT</td>
    <td>-</td>
</tr>
<tr>
    <td>eps</td>
    <td>属性</td>
    <td>用于防止产生除0的偏移。</td>
    <td>FLOAT</td>
    <td>-</td>
</tr>
<tr>
    <td>batch_mean</td>
    <td>输出</td>
    <td>表示全局批均值，对应公式中的batchMean。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
</tr>
<tr>
    <td>batch_invstd</td>
    <td>输出</td>
    <td>表示标准差倒数，对应公式中的batchInvstd。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
</tr>
<tr>
    <td>mean</td>
    <td>输出</td>
    <td>表示更新后的均值，对应公式中的runningMean。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
</tr>
<tr>
    <td>variance</td>
    <td>输出</td>
    <td>表示更新后的方差，对应公式中的runningVar。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
</tr>
</tbody></table>

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：参数`totalSum`、`totalSquareSum`、`mean`、`variance`、`batchMean`、`batchInvstd`的数据类型不支持BFLOAT16。


## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_sync_batch_norm_gather_stats](../sync_batch_norm_gather_stats/examples/test_aclnn_sync_batch_norm_gather_stats.cpp) | 通过[aclnnSyncBatchNormGatherStats](../sync_batch_norm_gather_stats/docs/aclnnSyncBatchNormGatherStats.md)接口方式调用SyncBatchNormGatherStats算子。 |
| 图模式 | [test_geir_sync_batch_norm_gather_stats](../sync_batch_norm_gather_stats/examples/test_geir_sync_batch_norm_gather_stats.cpp) | 通过[算子IR](op_graph/sync_batch_norm_gather_stats_proto.h)构图方式调用SyncBatchNormGatherStats算子。         |