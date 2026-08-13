# FakeQuantWithMinMaxVarsGradient

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

- 算子功能：FakeQuantWithMinMaxVars的反向梯度算子，用于量化感知训练（QAT）的反向传播。通过Nudge后的`nudgedMin`/`nudgedMax`构建0/1 mask，对梯度进行乘法门控（Straight-Through Estimator），同时计算`min`/`max`的越界梯度之和。

- 计算公式：

  **Nudge预计算（Device侧，与前向FakeQuantWithMinMaxVars相同）：**

  - $qMin = narrow\_range ? 1 : 0$
  - $qMax = 2^{num\_bits} - 1$
  - $scale = (max - min) / (qMax - qMin)$
  - $nudgedZeroPoint = round(qMin - min / scale)$，裁剪至 $[qMin, qMax]$
  - $nudgedMin = (qMin - nudgedZeroPoint) \times scale$
  - $nudgedMax = (qMax - nudgedZeroPoint) \times scale$

  注：上述Nudge中间值在Kernel Device侧从`min`/`max` tensor输入计算，传递到kernel的仅为`numBits`和`narrowRange`属性。

  **梯度计算（kernel端）：**

  $$mask = \begin{cases} 1, & nudgedMin \le x \le nudgedMax \\ 0, & \text{otherwise} \end{cases}$$

  $$backprops\_wrt\_x = gradients \times mask$$

  $$backprop\_wrt\_min = \sum_{i: x_i < nudgedMin} gradients_i$$

  $$backprop\_wrt\_max = \sum_{i: x_i > nudgedMax} gradients_i$$

  - 当`x`为NaN时，`mask = 0`（NaN比较结果为false）；当`gradients`为NaN时，NaN通过乘法自然传播（IEEE754: NaN × 0/1 = NaN）。
  - 乘法后对所有元素执行sign-bit OR（`vOut_u32 |= vG_u32 & 0x80000000`），修复Ascend Mul可能丢失-0符号位的问题；对正常数值和NaN该操作幂等无副作用。
  - 当`|min| + |max| < 1e-10`（退化）时：`backprops_wrt_x = gradients`（全透传），`backprop_wrt_min = 0`，`backprop_wrt_max = 0`。

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
      <td><ul><li>表示上游梯度Tensor，对应公式中的gradients；</li><li>shape与输入x完全一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td><ul><li>表示前向输入Tensor，用于构建mask判断x是否在量化范围内；</li><li>shape与gradients完全一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min</td>
      <td>输入</td>
      <td><ul><li>表示量化范围的最小值（标量tensor）；</li><li>shape必须为(1,)；</li><li>作为可训练变量参与梯度更新。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max</td>
      <td>输入</td>
      <td><ul><li>表示量化范围的最大值（标量tensor）；</li><li>shape必须为(1,)；</li><li>作为可训练变量参与梯度更新。</li></ul></td>
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
      <td><ul><li>表示x的梯度，经mask门控后的gradients；</li><li>shape和输入x一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprop_wrt_min</td>
      <td>输出</td>
      <td><ul><li>表示min的梯度，即x &lt; nudgedMin的元素对应的gradients之和；</li><li>shape固定为(1,)。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprop_wrt_max</td>
      <td>输出</td>
      <td><ul><li>表示max的梯度，即x &gt; nudgedMax的元素对应的gradients之和；</li><li>shape固定为(1,)。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入`gradients`、`x`的数据类型仅支持FLOAT32，shape必须完全一致（rank与每维大小均相同）。
- 输入`min`、`max`的数据类型为FLOAT32，shape必须为(1,)。
- `num_bits`取值范围为[2, 16]。
- 当`|min| + |max| < 1e-10`时走退化路径：`backprops_wrt_x`直接透传`gradients`，`backprop_wrt_min`和`backprop_wrt_max`输出为0。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | - | 通过GE IR构图方式调用FakeQuantWithMinMaxVarsGradient算子（[算子IR定义](op_graph/fake_quant_with_min_max_vars_gradient_proto.h)）。min/max作为标量tensor输入（可训练变量），num_bits/narrow_range作为属性。|
