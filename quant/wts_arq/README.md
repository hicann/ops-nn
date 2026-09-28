# WtsARQ

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

WtsARQ（Weights Adaptive Range Quantization）用于 AMCT 量化感知训练（QAT）场景：根据输入权重 `w` 及其量化范围下界 `w_min`、上界 `w_max` 自适应计算量化 scale，对 `w` 做 round-to-nearest（half-to-even）伪量化到 int8 栅格后再反量化回浮点，输出 `y`。算子以 GE IR 图模式交付，无 aclnn 接口；TensorFlow 等前端经框架插件映射为同一 IR 后调用。

计算流程（先做范围修正，再计算 scale）：

1. 范围修正：$w_{min} = min(w_{min}, 0)$，$w_{max} = max(w_{max}, 0)$
2. `offset_flag=false`（默认，非对称 abs-max）：$scale = max(|w_{min}| / 128,\ w_{max} / 127)$
3. `offset_flag=true`（对称带偏移）：$scale = w_{max}/255 - w_{min}/255$，并计算 $offset = -rint(w_{min}/scale) - 128$
4. 除零保护：$scale < 1.1920929 \times 10^{-7}$（float32 eps）时取 $scale = 1.0$
5. 伪量化（`rint` 为 round-to-nearest half-to-even）：$q = clamp(rint(w/scale) + offset,\ -128,\ 127) - offset$，`offset_flag=false` 时 $offset = 0$
6. 反量化：$y = q \times scale$

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
<td>w</td>
<td>输入</td>
<td>待伪量化的权重张量，是范围修正与量化公式的计算对象。</td>
<td>float16、float32</td>
<td>ND</td>
</tr>
<tr>
<td>w_min</td>
<td>输入</td>
<td>权重范围下界，按受限广播参与计算：先执行 w_min = min(w_min, 0)。</td>
<td>float16、float32</td>
<td>ND</td>
</tr>
<tr>
<td>w_max</td>
<td>输入</td>
<td>权重范围上界，按受限广播参与计算：先执行 w_max = max(w_max, 0)。</td>
<td>float16、float32</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>输出</td>
<td>伪量化并反量化后的权重，shape 与 dtype 和 w 相同。</td>
<td>float16、float32</td>
<td>ND</td>
</tr>
<tr>
<td>num_bits</td>
<td>可选属性</td>
<td>量化位宽，当前仅支持 8，默认值为 8。</td>
<td>int64</td>
<td>-</td>
</tr>
<tr>
<td>offset_flag</td>
<td>可选属性</td>
<td>是否使用 offset 的对称量化分支，默认值为 false；为 true 时按 w_max/255 - w_min/255 计算 scale。</td>
<td>bool</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- 输入与输出均为 ND 格式；非 ND 存储格式在 tiling 阶段被拒绝。
- `w`、`w_min`、`w_max`、`y` 的 dtype 必须一致，仅支持 float16 或 float32。
- `w` 的 rank 支持 0~8；rank=0（标量）按单元素张量处理。
- `w_min` 与 `w_max` 的 rank 必须等于 `w` 的 rank，并且两者 shape 完全相同；每一维必须等于 `w` 对应维或为 1（受限广播）。
- `w` 的元素总数不得超过 2^31；空 Tensor（存在零长度维）输出同样为空，不写回数据。
- 动态 shape 场景下 InferShape 支持未知维 `-1` 与未知 rank `-2`；其中可静态判定的非法维度关系会被拒绝，tiling 阶段要求所有维度已具体化。
- `num_bits` 仅支持 8，其他取值被拒绝。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| 图模式调用 | [test_geir_wts_arq](./examples/test_geir_wts_arq.cpp) | 通过[算子IR](./op_graph/wts_arq_proto.h)构图方式调用WtsARQ算子。|
