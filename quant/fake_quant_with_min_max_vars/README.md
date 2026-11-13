# FakeQuantWithMinMaxVars

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：对输入`x`进行per-tensor假量化（Fake Quantization），通过Nudge算法将`min`/`max`调整为量化步长的整数倍后，执行量化-反量化操作。`min`/`max`作为可训练变量（tensor输入）参与梯度更新，适用于量化感知训练（QAT）。对应的反向梯度算子为FakeQuantWithMinMaxVarsGradient。

- 计算公式：

  **Nudge预计算（kernel端，从`min`/`max` tensor输入计算）：**

  - $qMin = narrow\_range ? 1 : 0$
  - $qMax = 2^{num\_bits} - 1$
  - $scale = (max - min) / (qMax - qMin)$
  - $invScale = (qMax - qMin) / (max - min)$（独立除法计算，不复用 $1/scale$）
  - $nudgedZeroPoint = round(qMin - min / scale)$，裁剪至 $[qMin, qMax]$
  - $nudgedMin = (qMin - nudgedZeroPoint) \times scale$
  - $nudgedMax = (qMax - nudgedZeroPoint) \times scale$

  **前向计算（kernel端）：**

  $$
  clamped = clamp(x, nudgedMin, nudgedMax)
  $$

  $$
  y = floor(clamped \times invScale + 0.5) \times scale
  $$

  - 当 $min == max$ 时，输出全零（退化处理）。

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
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
</tr>
</thead>
<tbody>
<tr>
<td>x</td>
<td>输入</td>
<td>表示算子输入的Tensor，对应公式中的x；rank范围[1, 8]；shape与输出y一致。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>min</td>
<td>输入</td>
<td>表示量化范围的最小值（标量tensor），对应公式中的min；shape为(1,)；作为可训练变量参与梯度更新。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>max</td>
<td>输入</td>
<td>表示量化范围的最大值（标量tensor），对应公式中的max；shape为(1,)；作为可训练变量参与梯度更新。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>num_bits</td>
<td>可选属性</td>
<td>表示量化位宽，对应公式中的num_bits；取值范围[2, 16]；默认值为8。</td>
<td>INT</td>
<td>-</td>
</tr>
<tr>
<td>narrow_range</td>
<td>可选属性</td>
<td>表示是否使用窄量化范围；true时qMin=1，false时qMin=0；默认值为false。</td>
<td>BOOL</td>
<td>-</td>
</tr>
<tr>
<td>y</td>
<td>输出</td>
<td>表示假量化的计算输出，对应公式中的y；shape和输入x一致。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
</tbody>
</table>

## 约束说明

- 输入`x`与输出`y`的数据类型仅支持FLOAT32，shape必须完全一致。
- 输入`min`、`max`的数据类型为FLOAT32，shape必须为(1,)；当`min`/`max`的shape不为(1,)（含空张量，如shape为(0,)）时，算子在tiling阶段校验并拒绝执行（返回错误）。`x`的空维度属合法输入，空输入产空输出。
- `x`的rank范围为[1, 8]。
- `num_bits`取值范围为[2, 16]。
- 当`min == max`时，输出全零（退化处理）。
- 异常场景：对违反输入约束的场景（如数据类型不在支持列表、`min`/`max`的shape非法或为空张量、`num_bits`取值超出[2, 16]、数据格式不支持、缺少必需输入等），其中`min`/`max`的形状约束由算子在tiling阶段强制校验并拒绝执行，其余违反场景的处置（拒绝执行、报错或未定义的计算结果）与具体错误码由执行环境/框架决定，不承诺精确的处理行为与错误码。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | - | 通过GE IR构图方式调用FakeQuantWithMinMaxVars算子（[算子IR定义](op_graph/fake_quant_with_min_max_vars_proto.h)）。`min`/`max`作为标量tensor输入（可训练变量），`num_bits`/`narrow_range`作为属性。TensorFlow插件将TF的FakeQuantWithMinMaxVars op直接映射到本算子。 |
