# FusedCrossEntropyLossWithMaxSum

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     x    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：本算子是词汇表并行场景下交叉熵计算模块的一部分，解决超大规模词汇表下的显存和计算效率问题，当前部分为计算loss与softMax的结果。
- 计算公式：

    $$
    lossOut = log(sum\_exp\_logits) - predicted\_logits
    $$

    $$
    softMaxOutOptional = exp(vocab\_parallel\_logits -logits\_max.unsqueeze(dim = -1)) \ sum\_exp\_logits.unsqueeze(dim = -1)
    $$


## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 502px">
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
      <td>logitsMax</td>
      <td>输入</td>
      <td>matmul计算后各行的最大值，公式中的logitsMax。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sumExpLogits</td>
      <td>输入</td>
      <td>matmul计算结果与其各行的最大值作差后exp的结果。公式中的sumExpLogits。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>predictedLogits</td>
      <td>输入</td>
      <td>表示matmul计算结果与其各行的最大值作差后maskedTargetOut筛选后的结果。公式中的predictedLogits。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputOptional</td>
      <td>输入</td>
      <td>matmul输入左矩阵。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weightOptional</td>
      <td>输入</td>
      <td>matmul输入右矩阵。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vocabParallelLogitsOptional</td>
      <td>输入</td>
      <td>matmul计算结果。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>labelSmoothing</td>
      <td>属性</td>
      <td>标签平滑系数，用于缓解过拟合。</td>
      <td>float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>lossOut</td>
      <td>输出</td>
      <td>中间变量。公式中的loss。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softMaxOutOptional</td>
      <td>输出</td>
      <td>中间变量。公式中的vocabParallelLogits。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_fused_cross_entropy_loss_with_max_sum](examples/test_aclnn_fused_cross_entropy_loss_with_max_sum.cpp) | 通过[aclnnFusedCrossEntropyLossWithMaxSum](docs/aclnnFusedCrossEntropyLossWithMaxSum.md)接口方式调用FusedCrossEntropyLossWithMaxSum算子。 |