# NllLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列 / Atlas A3 推理系列</term>   |     √    |
|  <term>Atlas A2 训练系列 / Atlas A2 推理系列</term>     |     √    |

## 功能说明

- 算子功能：计算负对数似然损失（Negative Log Likelihood Loss），常用于配合 `LogSoftmax` 完成多分类训练。

- 计算公式：

  对每个样本 `i`，设其目标类别为 `target_i`。当 `target_i` 等于 `ignore_index` 时该样本不参与计算，否则：

  $$
  loss_i = -weight[target_i] * x[i, target_i]
  $$

  当 `reduction` 为 `none` 时，逐样本输出：

  $$
  y_i = loss_i
  $$

  当 `reduction` 为 `sum` 时：

  $$
  y = \sum_i loss_i
  $$

  当 `reduction` 为 `mean` 时：

  $$
  y = \frac{\sum_i loss_i}{\sum_i weight[target_i]}
  $$

  同时输出参与计算样本的权重之和：

  $$
  total\_weight = \sum_i weight[target_i]
  $$

  其中当未提供 `weight` 时，各类别权重按 `1` 处理。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>x</td>
      <td>输入</td>
      <td>公式中的输入x，最后一维为类别数C，其余维度展平为样本数N。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td>公式中的输入target，每个样本的目标类别索引，元素个数为样本数N。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>可选输入</td>
      <td>公式中的输入weight，各类别的权重，长度为C；未传入时权重按1处理。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>可选属性</td>
      <td>公式中的输入reduction，指定损失函数的计算方式，支持 'none' | 'mean' | 'sum'。'none' 表示不应用归约，'mean' 表示损失的加权平均，'sum' 表示损失求和。默认为 'mean'。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ignore_index</td>
      <td>可选属性</td>
      <td>公式中的输入ignore_index，指定被忽略且不参与损失计算的目标类别值。默认为 -100。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出y。reduction 为 'none' 时形状与 target 一致，否则为标量。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>total_weight</td>
      <td>输出</td>
      <td>公式中的输出total_weight，参与计算样本的权重之和。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `target` 的取值需落在 `[0, C)` 区间内，或等于 `ignore_index`。
- `x` 的最后一维为类别数 `C`，`target` 的元素个数为样本数 `N`。
- 若传入 `weight`，其长度需与类别数 `C` 保持一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| aclnn调用 | [test_aclnn_nll_loss](./examples/test_aclnn_nll_loss.cpp) | 通过[aclnnNLLLoss](./docs/aclnnNLLLoss.md)接口方式调用NllLoss算子。 |
| aclnn调用 | - | 通过[aclnnNLLLoss2d](./docs/aclnnNLLLoss2d.md)接口方式调用NllLoss算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| weixin_45448057 | 个人开发者 | NllLoss | 2026/07/14 | NllLoss算子适配开源仓 |
