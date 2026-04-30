# L1LossGrad

## 产品支持情况
|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明
- 算子功能：计算 L1 Loss 的反向梯度。
- 计算公式：
$$
dx = \text{grads} \times \text{sign}(\text{predict} - \text{label}) \times cof
$$
其中：
- 当 `reduction="mean"` 时，`cof = 1.0 / (shape_size)`
- 当 `reduction="none"` 或 `reduction="sum"` 时，`cof = 1.0`

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
      <td>grads</td>
      <td>输入</td>
      <td>反向传播的梯度输入。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>predict</td>
      <td>输入</td>
      <td>前向传播的预测值输入。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>输入</td>
      <td>目标标签输入。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>reduction</td>
      <td>属性</td>
      <td>规约模式，可选"none"、"mean"或"sum"。默认为"mean"。</td>
      <td>string</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>梯度计算结果输出。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明
- 输入张量 `grads`、`predict` 和 `label` 的形状必须完全一致，不支持广播机制。
- 输入输出数据类型必须一致。
- 数据格式仅支持 ND 格式。

## 调用说明
| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_l1_loss_grad.cpp](./examples/test_aclnn_l1_loss_grad.cpp) | 通过[aclnnL1LossBackward](./docs/aclnnL1LossBackward.md)接口方式调用L1LossGrad算子。 |

## 贡献说明
| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| 赵昊 | 浙江工业大学-智能计算研究所 | L1LossGrad | 2026/04/16 | L1LossGrad算子适配开源仓 |
```