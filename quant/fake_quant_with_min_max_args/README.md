# FakeQuantWithMinMaxArgs

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

- 算子功能：对输入 `x` 进行 per-tensor 假量化（Fake Quantization），通过 Nudge 算法将 `min`/`max` 调整为量化步长的整数倍后，执行量化-反量化操作。

- 计算公式：

  **Nudge 预计算（host 端）：**

  - $qMin = narrow\_range ? 1 : 0$
  - $qMax = 2^{num\_bits} - 1$
  - $scale = (max - min) / (qMax - qMin)$
  - $scaleInv = (qMax - qMin) / (max - min)$（必须独立除法计算，不能用 $1/scale$，精度保障）
  - $nudgedZeroPoint = round(qMin - min / scale)$，裁剪至 $[qMin, qMax]$
  - $nudgedMin = (qMin - nudgedZeroPoint) \times scale$
  - $nudgedMax = (qMax - nudgedZeroPoint) \times scale$
  - $quantZero = floor(-nudgedMin \times scaleInv + 0.5)$（必须基于 nudgedMin 重算，不能复用 nudgedZeroPoint，精度保障）

  **前向计算（kernel 端）：**

  $$
  clamped = clamp(x, nudgedMin, nudgedMax)
  $$

  $$
  q\_offset = floor((clamped - nudgedMin) \times scaleInv + (0.5 - quantZero))
  $$

  $$
  y = q\_offset \times scale
  $$

  - 当输入 `x` 为 NaN 时，NaN 直通到输出 `y`（NaN passthrough）。

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
      <td>x</td>
      <td>输入</td>
      <td><ul><li>表示算子输入的 Tensor，对应公式中的 x；</li><li>shape 与输出 y 一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min</td>
      <td>可选属性</td>
      <td><ul><li>表示量化范围的最小值，对应公式中的 min；</li><li>必须小于 max；</li><li>缺省值为 -6.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max</td>
      <td>可选属性</td>
      <td><ul><li>表示量化范围的最大值，对应公式中的 max；</li><li>必须大于 min；</li><li>缺省值为 6.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_bits</td>
      <td>可选属性</td>
      <td><ul><li>表示量化位宽，对应公式中的 num_bits；</li><li>取值范围 [2, 16]；</li><li>缺省值为 8。</li></ul></td>
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
      <td><ul><li>表示假量化的计算输出，对应公式中的 y；</li><li>shape 和输入 x 一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入 `x` 与输出 `y` 的数据类型仅支持 FLOAT32，shape 必须完全一致。
- `min` 必须小于 `max`。
- `num_bits` 取值范围为 [2, 16]。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_fake_quant_with_min_max_args](examples/test_geir_fake_quant_with_min_max_args.cpp) | 通过 GE IR 构图方式调用 FakeQuantWithMinMaxArgs 算子（per-tensor min/max/num_bits/narrow_range 标量属性）。 |