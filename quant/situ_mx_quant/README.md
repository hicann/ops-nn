# SituMxQuant

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

- 算子功能：将Situ激活函数与动态MX（Microscaling）量化融合为一个算子。

- 计算公式：

  1. Situ激活：

     $$
     situ_a = \beta \times \tanh(gate / \beta) \times sigmoid(gate)
     $$

     当linear_beta > 0时：

     $$
     up = linear\_beta \times \tanh(up / linear\_beta)
     $$

     $$
     situOut = situ_a \times up
     $$

     其中，当activate_left为true时，gate取x的前半部分，up取后半部分；当activate_left为false时，gate取x的后半部分，up取前半部分。

  2. MX量化（OCP算法）：

     $$
     shared\_exp = floor(log2(max(|situOut_i|))) - emax
     $$

     $$
     y\_scale = 2^{shared\_exp}  (E8M0)
     $$

     $$
     y = cast\_to\_fp8(situOut / y\_scale)
     $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 951px"><colgroup>
  <col style="width: 121px">
  <col style="width: 144px">
  <col style="width: 313px">
  <col style="width: 257px">
  <col style="width: 116px">
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
      <td>输入待处理的数据，公式中的x。最后一维需要是2的倍数。输入不支持包含±inf或nan。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>属性</td>
      <td>Situ激活的beta参数，公式中的β。不能为0。默认1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linear_beta</td>
      <td>属性</td>
      <td>Situ激活的linear_beta参数，公式中的linear_beta。当值≤0时不启用。默认0.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>属性</td>
      <td>表示gate取x的前半部分还是后半部分，公式中的activate_left。默认false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>表示量化轴，公式中max和scale的计算轴。当前仅支持-1。默认-1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>属性</td>
      <td>表示输出y的数据类型，对应公式中cast_to_fp8的目标类型：36=FLOAT8_E4M3FN，35=FLOAT8_E5M2。默认36。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>属性</td>
      <td>表示量化舍入模式，公式中cast的舍入方式。支持"rint"、"round"、"floor"。FP8输出仅支持"rint"。默认"rint"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>量化后的输出，公式中的y。</td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_scale</td>
      <td>输出</td>
      <td>MX量化的scale（E8M0格式），公式中的y_scale。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x的最后一维需要是2的倍数。
- x的维数必须大于等于1维。
- axis当前仅支持-1（尾轴量化）。
- beta参数不能为0。
- dst_type支持36（FLOAT8_E4M3FN）或35（FLOAT8_E5M2）。
- round_mode必须为"rint"。
- 关于y_scale的shape约束说明如下：
  - H = x.shape[-1] / 2
  - scaleNum = ceil(H / 64)
  - y_scale.shape = x.shape[:-1] + [scaleNum, 2]

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_situ_mx_quant](./examples/arch35/test_aclnn_situ_mx_quant.cpp) | 通过[aclnnSituMxQuant](./docs/aclnnSituMxQuant.md)接口方式调用SituMxQuant算子。    |
| 图模式调用 | - | 通过[算子IR](./op_graph/situ_mx_quant_proto.h)构图方式调用SituMxQuant算子。 |
