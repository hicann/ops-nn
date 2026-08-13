# FakeQuantWithMinMaxVarsPerChannel

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

- 算子功能：对输入`x`进行per-channel假量化（Fake Quantization），沿最后一轴（channel轴）的每个通道使用各自的`min[c]`/`max[c]`独立执行Nudge量化-反量化操作。与per-tensor版本（FakeQuantWithMinMaxVars）的区别在于每个通道拥有独立的量化参数，能更好地适应不同通道的数值范围差异，显著提升权重量化精度。

- 计算公式：

  **Per-Channel Nudge预计算（kernel端，每个通道独立）：**

  - $qMin = narrow\_range ? 1 : 0$
  - $qMax = 2^{num\_bits} - 1$
  - $scale[c] = (max[c] - min[c]) / (qMax - qMin)$
  - $scaleInv[c] = (qMax - qMin) / (max[c] - min[c])$（必须独立除法计算，不能用 $1/scale$，精度保障）
  - $nudgedZeroPoint[c] = round(qMin - min[c] / scale[c])$，裁剪至 $[qMin, qMax]$
  - $nudgedMin[c] = (qMin - nudgedZeroPoint[c]) \times scale[c]$
  - $nudgedMax[c] = (qMax - nudgedZeroPoint[c]) \times scale[c]$

  **前向计算（kernel端，per-channel元素级）：**

  $$
  clamped = clamp(x[..., c], nudgedMin[c], nudgedMax[c])
  $$

  $$
  y[..., c] = floor((clamped - nudgedMin[c]) \times scaleInv[c] + 0.5) \times scale[c] + nudgedMin[c]
  $$

  - 当某通道 $|min[c]| + |max[c]| = 0$ 时，该通道输出全零（退化处理）。
  - 当输入`x`为NaN时，NaN经clamp/round后传播到输出`y`。

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
      <td>x</td>
      <td>输入</td>
      <td><ul><li>表示算子输入的Tensor，最后一轴为channel C；</li><li>rank范围[1, 8]；</li><li>shape与输出y一致。</li></ul></td>
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
      <td>y</td>
      <td>输出</td>
      <td><ul><li>表示per-channel假量化的计算输出；</li><li>shape和输入x完全一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入`x`与输出`y`的数据类型仅支持FLOAT32，shape必须完全一致。
- 输入`min`、`max`必须为1-D Tensor（rank=1），且长度等于`x`最后一维的大小C。
- `min`和`max`的长度必须一致。
- `num_bits`取值范围为[2, 16]。
- `x`的rank范围必须为[1, 8]。
- 当某通道`|min[c]| + |max[c]| = 0`时，该通道输出全零。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | - | 通过GE IR构图方式调用FakeQuantWithMinMaxVarsPerChannel算子（[算子IR定义](op_graph/fake_quant_with_min_max_vars_per_channel_proto.h)）。min/max作为1-D tensor输入（per-channel可训练变量），num_bits/narrow_range作为属性。TF插件会将输入/输出format设为NHWC，使最后一轴对应channel。|
