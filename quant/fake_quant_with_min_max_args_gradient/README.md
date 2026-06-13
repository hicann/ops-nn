# FakeQuantWithMinMaxArgsGradient

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

- 算子功能：FakeQuantWithMinMaxArgs的反向梯度算子，通过Nudge后的`nudgedMin`/`nudgedMax`构建0/1mask，对梯度进行乘法门控。

- 计算公式：

  **Nudge预计算（host端，与FakeQuantWithMinMaxArgs相同）：**

  - $qMin = narrow\_range ? 1 : 0$
  - $qMax = 2^{num\_bits} - 1$
  - $scale = (max - min) / (qMax - qMin)$
  - $nudgedZeroPoint = round(qMin - min / scale)$，裁剪至 $[qMin, qMax]$
  - $nudgedMin = (qMin - nudgedZeroPoint) \times scale$
  - $nudgedMax = (qMax - nudgedZeroPoint) \times scale$

  注：上述Nudge中间值仅用于host端预计算，传递到kernel的仅为`nudgedMin`和`nudgedMax`。

  **梯度计算（kernel端）：**

  $$
  mask = \begin{cases} 1, & nudgedMin \le x \le nudgedMax \\ 0, & \text{otherwise} \end{cases}
  $$

  $$
  y = gradients \times mask
  $$

  - 当`x`为NaN时，`mask = 0`（NaN比较结果为false）；当`gradients`为NaN时，NaN通过乘法自然传播（IEEE754: NaN × 0/1 = NaN）。
  - 乘法后对所有元素执行sign-bit OR（vOut_u32 |= vG_u32 & 0x80000000），修复Ascend Mul可能丢失-0符号位的问题；对正常数值和NaN该操作幂等无副作用。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
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
      <td>gradients</td>
      <td>输入</td>
      <td><ul><li>表示上游梯度Tensor，对应公式中的gradients；</li><li>shape与输入x、输出y一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td><ul><li>表示前向输入Tensor，用于构建mask判断x是否在量化范围内；</li><li>shape与gradients、输出y一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min</td>
      <td>可选属性</td>
      <td><ul><li>表示量化范围的最小值；</li><li>必须小于max；</li><li>缺省值为-6.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max</td>
      <td>可选属性</td>
      <td><ul><li>表示量化范围的最大值；</li><li>必须大于min；</li><li>缺省值为6.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_bits</td>
      <td>可选属性</td>
      <td><ul><li>表示量化位宽；</li><li>取值范围 [2, 16]；</li><li>缺省值为8。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>narrow_range</td>
      <td>可选属性</td>
      <td><ul><li>表示是否使用窄量化范围；</li><li>true时qMin=1，false时qMin=0；</li><li>缺省值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>表示梯度计算输出，对应公式中的y；</li><li>shape和输入x一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入`gradients`、`x`与输出`y`的数据类型仅支持FLOAT32，shape必须完全一致。
- `min`必须小于`max`。
- `num_bits`取值范围为[2, 16]。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_fake_quant_with_min_max_args_gradient](examples/test_geir_fake_quant_with_min_max_args_gradient.cpp) | 通过GE IR构图方式调用FakeQuantWithMinMaxArgsGradient算子（per-tensor min/max/num_bits/narrow_range标量属性）。 |