# MseLoss

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- 算子功能：计算输入 `predict` 和目标 `label` 逐元素平方误差，并根据 `reduction` 输出不缩减、求和或求均值结果。

- 计算公式：

  $$
  loss_i = (predict_i - label_i)^2
  $$

  $$
  y =
  \begin{cases}
  loss_i, & reduction = none \\
  \sum_{i=0}^{N-1} loss_i, & reduction = sum \\
  \frac{1}{N}\sum_{i=0}^{N-1} loss_i, & reduction = mean
  \end{cases}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>predict</td>
      <td>输入</td>
      <td>预测值张量，公式中的 predict。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>输入</td>
      <td>目标值张量，公式中的 label。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>属性</td>
      <td>损失缩减方式，可选值为 none、sum、mean，默认值为 mean。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出 loss。reduction 为 none 时 shape 与输入一致；sum 或 mean 时为单元素输出。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `predict`、`label`、`y` 的数据类型必须一致。
- `predict` 与 `label` 的 shape 必须一致。
- `reduction` 仅支持 `none`、`sum`、`mean`。
- 当前实现不支持输入广播。
- 当前实现仅支持 ND 数据格式。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| aclnn调用 | [test_aclnn_mse_loss.cpp](./examples/test_aclnn_mse_loss.cpp) | 通过 `aclnnMseLossGetWorkspaceSize` 和 `aclnnMseLoss` 两段式接口调用 MseLoss 算子。 |

## 参考资源

- [MseLoss 算子设计文档](./docs/mse_loss_design.md)

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| ji096929 | 个人开发者 | MseLoss | 2026/07/09 | MseLoss算子适配开源仓 |
