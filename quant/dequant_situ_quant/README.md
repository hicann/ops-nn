# DequantSituQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：在Situ激活函数前后添加dequant和quant操作，实现x的DequantSituQuant计算。

- 计算公式：

  1. 根据输入数据类型x的不同，反量化路径不同：

     - INT8路径：

       $$
       dequantOut_i = cast\_to\_float(x_i) \times weight\_scale_i + bias_i
       $$

     - INT32路径：

       $$
       dequantOut_i = cast\_to\_float(x_i) \times weight\_scale_i \times activation\_scale_i + bias_i
       $$

     - BF16/FLOAT16路径（预反量化）：

       $$
       dequantOut_i = cast\_to\_float(x_i)
       $$

  2. Situ激活：

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

     其中，当activate_left为true时，gate取dequantOut的前半部分，up取后半部分；当activate_left为false时，gate取dequantOut的后半部分，up取前半部分。

  3. 量化：

     - static模式：

       $$
       out_i = trunc(situOut_i / quant\_scale_i + quant\_offset_i)
       $$

     - dynamic模式：

       $$
       scale_i = absmax(situOut_i) / 127
       $$

       $$
       out_i = trunc(situOut_i / scale_i)
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
      <td>输入待处理的数据，公式中的x。输入不支持包含±inf或nan。</td>
      <td>INT8、INT32、BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_scale</td>
      <td>输入</td>
      <td>反量化的weight scale，公式中的weight_scale。输入不支持包含±inf或nan。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activation_scale</td>
      <td>输入</td>
      <td>反量化的activation scale，公式中的activation_scale。输入不支持包含±inf或nan。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>反量化的bias，公式中的bias。输入不支持包含±inf或nan。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_scale</td>
      <td>输入</td>
      <td>量化的scale，公式中的quant_scale。输入不支持包含±inf或nan。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_offset</td>
      <td>输入</td>
      <td>量化的offset，公式中的quant_offset。输入不支持包含±inf或nan。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>输入</td>
      <td>MoE分组需要的group_index。输入不支持包含±inf或nan。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>属性</td>
      <td>Situ激活的beta参数，公式中的β。不能为0。默认4.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linear_beta</td>
      <td>属性</td>
      <td>Situ激活的linear_beta参数，公式中的linear_beta。当值≤0时不启用。默认25.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>属性</td>
      <td>表示gate取dequantOut的前半部分还是后半部分，公式中的activate_left。默认true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_type</td>
      <td>属性</td>
      <td>表示量化模式，对应公式中的static/dynamic模式。支持"static"和"dynamic"。默认"dynamic"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>量化后的输出，公式中的out。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_scale</td>
      <td>输出</td>
      <td>动态量化的scale，公式中的scale（仅dynamic模式有意义）。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x的最后一维需要是2的倍数。
- 当x的数据类型为INT8时，x维度≥2维；当x的数据类型为INT32/BF16/FLOAT16时，x维度为2维。
- beta参数不能为0。
- INT8路径：必须提供weight_scale，禁止activation_scale和group_index。
- INT32：必须提供weight_scale和activation_scale，禁止quant_scale和quant_offset。
- BFLOAT16/FLOAT16：所有可选输入均不使用（预反量化模式）。
- 当quant_type为static时，quant_scale必须提供。
- 当quant_type为dynamic时，quant_scale可选（作为smoothScale使用）。
- 算子支持的输入张量的内存大小有上限，校验公式：weight_scale张量内存大小+bias张量内存大小+quant_scale张量内存大小+quant_offset张量内存大小 + （activation_scale张量内存大小 + y_scale张量内存大小）/40  + x张量最后一维H内存大小 * 10 < 192KB。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dequant_situ_quant](./examples/test_aclnn_dequant_situ_quant.cpp) | 通过[aclnnDequantSituQuant](./docs/aclnnDequantSituQuant.md)接口方式调用DequantSituQuant算子。    |
| 图模式调用 | - | 通过[算子IR](./op_graph/dequant_situ_quant_proto.h)构图方式调用DequantSituQuant算子。 |
