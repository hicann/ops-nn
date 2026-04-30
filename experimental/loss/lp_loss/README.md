# LpLoss

## 产品支持情况
|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品</term>     |     √    |

## 功能说明
- 算子功能：计算预测值与标签值之间的 Lp 范数损失。
- 计算公式：
$$
y_i = |predict_i - label_i|^p
$$
其中：
- 当 `reduction="none"` 时，输出为逐元素损失，`y` 的 shape 与输入一致。
- 当 `reduction="mean"` 时，输出为全部元素损失的平均值。
- 当 `reduction="sum"` 时，输出为全部元素损失的求和值。
- 当前仅支持 `p=1`。

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
      <td>预测值输入。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>输入</td>
      <td>标签值输入。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>属性</td>
      <td>Lp 范数的阶数，当前仅支持 `1`。</td>
      <td>int64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>属性</td>
      <td>规约模式，可选 `"none"`、`"mean"` 或 `"sum"`。默认为 `"mean"`。</td>
      <td>string</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>损失计算结果输出。当 `reduction="none"` 时 shape 与输入一致，否则输出标量。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明
- 输入张量 `predict` 和 `label` 的形状必须完全一致，不支持广播机制。
- 输入输出数据类型必须一致。
- 数据格式仅支持 ND 格式。
- 属性 `p` 当前仅支持 `1`。

## 调用说明
| 调用方式 | 调用样例                                                               | 说明                                                   |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------|
| aclnn调用 | [test_aclnn_lp_loss.cpp](./examples/test_aclnn_lp_loss.cpp) | 通过[aclnnLpLoss](./docs/aclnnLpLoss.md)接口方式调用LpLoss算子。 |

## 贡献说明
| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| 雷洋 | 浙工大 | LpLoss | 2025/01/21 | 新增LpLoss算子 |