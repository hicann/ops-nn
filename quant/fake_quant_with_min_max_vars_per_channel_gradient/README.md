# FakeQuantWithMinMaxVarsPerChannelGradient

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：FakeQuantWithMinMaxVarsPerChannel的反向梯度算子，用于per-channel量化感知训练（QAT）的反向传播。通过per-channel Nudge后的`nudgedMin[c]`/`nudgedMax[c]`构建0/1 mask，对梯度进行乘法门控（Straight-Through Estimator），同时计算`min`/`max`的per-channel越界梯度之和。

- 计算公式：

  **Per-Channel Nudge预计算（Device侧，与前向FakeQuantWithMinMaxVarsPerChannel相同）：**

  - $qMin = narrow\_range ? 1 : 0$
  - $qMax = 2^{num\_bits} - 1$
  - $scale[c] = (max[c] - min[c]) / (qMax - qMin)$
  - $nudgedZeroPoint[c] = round(qMin - min[c] / scale[c])$，裁剪至 $[qMin, qMax]$
  - $nudgedMin[c] = (qMin - nudgedZeroPoint[c]) \times scale[c]$
  - $nudgedMax[c] = (qMax - nudgedZeroPoint[c]) \times scale[c]$

  注：上述Nudge中间值在Kernel Device侧从`min`/`max` tensor输入按通道独立计算。

  **梯度计算（kernel端，per-channel）：**

  $$mask[..., c] = \begin{cases} 1, & nudgedMin[c] \le x[..., c] \le nudgedMax[c] \\ 0, & \text{otherwise} \end{cases}$$

  $$backprops\_wrt\_x[..., c] = gradients[..., c] \times mask[..., c]$$

  $$backprop\_wrt\_min[c] = \sum_{i: x_i < nudgedMin[c]} gradients_i$$

  $$backprop\_wrt\_max[c] = \sum_{i: x_i > nudgedMax[c]} gradients_i$$

  - 当某通道`x`为NaN时，`mask = 0`（NaN比较结果为false）；当`gradients`为NaN时，NaN通过乘法自然传播。
  - 当某通道`|min[c]| + |max[c]| = 0`（退化）时：`backprops_wrt_x`透传`gradients`，`backprop_wrt_min/max`为0。
  - backprop_wrt_min/max的跨行归约使用Kahan补偿求和，减少浮点累加误差。

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>gradients</td>
      <td>输入</td>
      <td><ul><li>表示上游梯度Tensor，最后一轴为channel C；</li><li>shape与输入x完全一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td><ul><li>表示前向输入Tensor，用于构建per-channel mask；</li><li>shape与gradients完全一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min</td>
      <td>输入</td>
      <td><ul><li>表示per-channel量化范围的最小值；</li><li>1-D Tensor，长度等于x最后一维C；</li><li>作为可训练变量参与梯度更新。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max</td>
      <td>输入</td>
      <td><ul><li>表示per-channel量化范围的最大值；</li><li>1-D Tensor，长度等于x最后一维C；</li><li>作为可训练变量参与梯度更新。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_bits</td>
      <td>可选属性</td>
      <td><ul><li>表示量化位宽；</li><li>取值范围[2, 16]；</li><li>默认值为8。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>narrow_range</td>
      <td>可选属性</td>
      <td><ul><li>表示是否使用窄量化范围；</li><li>true时qMin=1，false时qMin=0；</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>backprops_wrt_x</td>
      <td>输出</td>
      <td><ul><li>表示x的梯度，经per-channel mask门控后的gradients；</li><li>shape和输入x一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprop_wrt_min</td>
      <td>输出</td>
      <td><ul><li>表示min的per-channel梯度，即各通道x &lt; nudgedMin[c]的元素对应的gradients之和；</li><li>shape为[C]，与min一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprop_wrt_max</td>
      <td>输出</td>
      <td><ul><li>表示max的per-channel梯度，即各通道x &gt; nudgedMax[c]的元素对应的gradients之和；</li><li>shape为[C]，与max一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入`gradients`、`x`的数据类型仅支持FLOAT32，shape必须完全一致（rank与每维大小均相同）。
- 输入`min`、`max`必须为1-D Tensor（rank=1），且长度等于`x`最后一维的大小C。
- `min`和`max`的长度必须一致。
- `num_bits`取值范围为[2, 16]。
- `x`的rank范围必须为[1, 8]。
- 当某通道`|min[c]| + |max[c]| = 0`时，该通道`backprops_wrt_x`透传`gradients`，`backprop_wrt_min/max`为0。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | - | 通过GE IR构图方式调用FakeQuantWithMinMaxVarsPerChannelGradient算子（[算子IR定义](op_graph/fake_quant_with_min_max_vars_per_channel_gradient_proto.h)）。min/max作为1-D tensor输入（per-channel可训练变量），num_bits/narrow_range作为属性。TF插件会将所有输入/输出format设为NHWC。|
