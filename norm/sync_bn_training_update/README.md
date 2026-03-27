# SyncBNTrainingUpdate

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：SyncBNTrainingUpdate算子用于更新在BatchNormTraining过程中的全局的均值。


- 计算公式：

$$
runningMeanUpdate = (mean * momentum) + runningMean * (1 - momentum)
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
      <td>mean</td>
      <td>输入</td>
      <td>表示输入数据均值，对应公式中的`mean`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>running_mean</td>
      <td>输入</td>
      <td>表示计算过程中的均值，对应公式中的`runningMean`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>momentum</td>
      <td>属性</td>
      <td>runningMean的指数平滑参数,默认值0.1。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>running_mean_update</td>
      <td>输出</td>
      <td>更新后的均值，对应公式中的`runningMeanUpdate`。</td>
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
| aclnn接口  | [test_aclnn_BatchNormGatherStatsWithCounts](../sync_batch_norm_gather_stats_with_counts/examples/test_aclnn_BatchNormGatherStatsWithCounts.cpp) | 通过[aclnnBatchNormGatherStatsWithCounts](../sync_batch_norm_gather_stats_with_counts/docs/aclnnBatchNormGatherStatsWithCounts.md)接口方式调用SyncBNTrainingUpdate。 |
| 图模式 | [test_geir_sync_bn_training_update](../sync_bn_training_update/examples/test_geir_sync_bn_training_update.cpp) | 通过[算子IR](op_graph/sync_bn_training_update_proto.h)构图方式调用SyncBNTrainingUpdate算子。         |