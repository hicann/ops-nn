# SyncBatchNormBackwardReduce

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列 / Atlas A3 推理系列</term>   |     √    |
|  <term>Atlas A2 训练系列 / Atlas A2 推理系列</term>     |     √    |

## 功能说明

- 算子功能：同步 BatchNorm 反向阶段的按通道归约，在已聚合的 `sum_dy`、`sum_dy_dx_pad` 基础上，结合每通道 `mean`、`invert_std` 计算 `sum_dy_xmu` 与 `y`，为后续权重/偏置梯度提供输入。

- 计算公式：

  对每个通道 `i`，先计算中心化梯度求和：

  $$
  sum\_dy\_xmu_i = sum\_dy\_dx\_pad_i - mean_i \cdot sum\_dy_i
  $$

  再乘以标准差倒数得到输出：

  $$
  y_i = sum\_dy\_xmu_i \cdot invert\_std_i
  $$

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
      <td>sum_dy</td>
      <td>输入</td>
      <td>公式中的输入sum_dy，每通道回传梯度求和。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_dy_dx_pad</td>
      <td>输入</td>
      <td>公式中的输入sum_dy_dx_pad，每通道 dy 与 x 乘积的聚合值。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>公式中的输入mean，每通道均值。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>invert_std</td>
      <td>输入</td>
      <td>公式中的输入invert_std，每通道标准差倒数。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_dy_xmu</td>
      <td>输出</td>
      <td>公式中的输出sum_dy_xmu，每通道 dy 与中心化输入乘积之和。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出y，sum_dy_xmu 乘以标准差倒数的结果。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `sum_dy`、`sum_dy_dx_pad`、`mean`、`invert_std`、`sum_dy_xmu`、`y` 的 shape、dtype 需保持一致。
- 输出 `sum_dy_xmu`、`y` 的 shape、dtype 与 `sum_dy` 一致。
- 不支持广播。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| aclnn调用 | [test_aclnn_sync_batch_norm_backward_reduce](./examples/test_aclnn_sync_batch_norm_backward_reduce.cpp) | 通过[aclnnBatchNormReduceBackward](./docs/aclnnBatchNormReduceBackward.md)接口方式调用SyncBatchNormBackwardReduce算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| weixin_45448057 | 个人开发者 | SyncBatchNormBackwardReduce | 2026/07/14 | SyncBatchNormBackwardReduce算子适配开源仓 |
