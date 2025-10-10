# DynamicQuantV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：为输入张量进行per-token对称/非对称动态量化。在MOE场景下，每个专家的smoothScalesOptional是不同的，根据输入的groupIndexOptional进行区分。

- 计算公式：
  - 对称量化：
    - 若不输入smoothScalesOptional，则
      $$
        scaleOut=row\_max(abs(x))/127
      $$
      $$
        yOut=round(x/scaleOut)
      $$
    - 若输入smoothScalesOptional，则
      $$
        input = x\cdot smoothScalesOptional
      $$
      $$
        scaleOut=row\_max(abs(input))/127
      $$
      $$
        yOut=round(input/scaleOut)
      $$
  - 非对称量化：
    - 若不输入smoothScalesOptional，则
      $$
        scaleOut=(row\_max(x) - row\_min(x))/scale\_opt
      $$
      $$
        offset=offset\_opt-row\_max(x)/scaleOut
      $$
      $$
        yOut=round(x/scaleOut+offset)
      $$
    - 若输入smoothScalesOptional，则
      $$
        input = x\cdot smoothScalesOptional
      $$
      $$
        scaleOut=(row\_max(input) - row\_min(input))/scale\_opt
      $$
      $$
        offset=offset\_opt-row\_max(input)/scaleOut
      $$
      $$
        yOut=round(input/scaleOut+offset)
      $$
  其中row\_max代表每行求最大值，row_min代表每行求最小值。当输出yOut类型为INT8时，scale_opt为255.0，offset_opt为127.0；yOut类型为INT4时，scale_opt为15.0，offset_opt为7.0。

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
      <td><ul><li>算子输入的Tensor，对应公式中的x。</li><li>shape维度要大于1。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scales</td>
      <td>可选输入</td>
      <td><ul><li>算子输入的Tensor，对应公式中的smoothScalesOptional。</li><li>数据类型与x的数据类型保持一致。当group_index为空时，形状为1维。Dim[0]是x的最后一个维度；当group_index不为空时，形状为2维。Dim[0]是专家数(E)。E必须不大于1024。Dim[1]是x的最后一个维度。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>group_index</td>
      <td>可选输入</td>
      <td><ul><li>指定组的索引。</li><li>shape支持1D，为[E, ]，shape的第一维与smooth_scales形状的第一维相同。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>可选属性</td>
      <td><ul><li>指定输出y的数据类型。支持DT_INT4(29)、DT_INT8(2)、DT_FLOAT8_E5M2(35)、DT_FLOAT8_E4M3FN(36)、DT_HIFLOAT8(34)。</li><li>默认值为DT_INT8(2)。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_symmetrical</td>
      <td>可选属性</td>
      <td><ul><li>是否使用对称量化。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>可选属性</td>
      <td><ul><li>指定量化模式。支持"pertoken"、"pertensor"。</li><li>默认值为"pertoken"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>量化后的输出张量，对应公式中的yOut。</li><li>shape与输入x的保持一致。当y的dtype为int4，x的最后一个维度必须能被2整除。</li></ul></td>
      <td>INT8、INT4、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输出</td>
      <td><ul><li>量化后的缩放系数，对应公式中的scaleOut。</li><li>当quant_mode为"pertoken"时，形状与x去掉最后一个维度后的形状相同。当quant_mode为"pertensor"时，形状为(1,)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输出</td>
      <td><ul><li>非对称量化使用的offset，对应公式中的offset。</li><li>shape与scale的shape保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：输出`y`的数据类型仅支持INT8、INT4。

## 约束说明

- E不应大于x去掉最后一个维度后的维度的乘积结果(S)。
- `group_index`的值应递增，范围从0到S。最后一个值应为S。否则结果将无意义。

