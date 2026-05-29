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

- 算子功能：FakeQuantWithMinMaxArgs 的反向梯度算子，通过 Nudge 后的 `nudgedMin`/`nudgedMax` 构建 0/1 mask，对梯度进行乘法门控。

- 计算公式：

  **Nudge 预计算（host 端，与 FakeQuantWithMinMaxArgs 相同）：**

  - $qMin = narrow\_range ? 1 : 0$
  - $qMax = 2^{num\_bits} - 1$
  - $scale = (max - min) / (qMax - qMin)$
  - $nudgedZeroPoint = round(qMin - min / scale)$，裁剪至 $[qMin, qMax]$
  - $nudgedMin = (qMin - nudgedZeroPoint) \times scale$
  - $nudgedMax = (qMax - nudgedZeroPoint) \times scale$

  注：上述 Nudge 中间值仅用于 host 端预计算，传递到 kernel 的仅为 `nudgedMin` 和 `nudgedMax`。

  **梯度计算（kernel 端）：**

  $$
  mask = \begin{cases} 1, & nudgedMin \le x \le nudgedMax \\ 0, & \text{otherwise} \end{cases}
  $$

  $$
  y = gradients \times mask
  $$

  - 当 `x` 为 NaN 时，`mask = 0`（NaN 比较结果为 false）；当 `gradients` 为 NaN 时，NaN 通过乘法自然传播（IEEE754: NaN × 0/1 = NaN）。
  - 乘法后对所有元素执行 sign-bit OR（vOut_u32 |= vG_u32 & 0x80000000），修复 Ascend Mul 可能丢失 -0 符号位的问题；对正常数值和 NaN 该操作幂等无副作用。

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
      <td><ul><li>表示上游梯度 Tensor，对应公式中的 gradients；</li><li>shape 与输入 x、输出 y 一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td><ul><li>表示前向输入 Tensor，用于构建 mask 判断 x 是否在量化范围内；</li><li>shape 与 gradients、输出 y 一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min</td>
      <td>可选属性</td>
      <td><ul><li>表示量化范围的最小值；</li><li>必须小于 max；</li><li>缺省值为 -6.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max</td>
      <td>可选属性</td>
      <td><ul><li>表示量化范围的最大值；</li><li>必须大于 min；</li><li>缺省值为 6.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_bits</td>
      <td>可选属性</td>
      <td><ul><li>表示量化位宽；</li><li>取值范围 [2, 16]；</li><li>缺省值为 8。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>narrow_range</td>
      <td>可选属性</td>
      <td><ul><li>表示是否使用窄量化范围；</li><li>true 时 qMin=1，false 时 qMin=0；</li><li>缺省值为 false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>表示梯度计算输出，对应公式中的 y；</li><li>shape 和输入 x 一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入 `gradients`、`x` 与输出 `y` 的数据类型仅支持 FLOAT32，shape 必须完全一致。
- `min` 必须小于 `max`。
- `num_bits` 取值范围为 [2, 16]。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_fake_quant_with_min_max_args_gradient](examples/test_geir_fake_quant_with_min_max_args_gradient.cpp) | 通过 GE IR 构图方式调用 FakeQuantWithMinMaxArgsGradient 算子（per-tensor min/max/num_bits/narrow_range 标量属性）。 |