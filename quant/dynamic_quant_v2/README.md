# DynamicQuantV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：为输入张量进行per-token对称/非对称动态量化。在MOE场景下，每个专家的smooth_scales是不同的，根据输入的group_index进行区分。

- 计算公式：
  - 对称量化：
    - 若不输入smooth_scales，则
      $$
        scale=row\_max(abs(x))/127
      $$
      $$
        y=round(x/scale)
      $$
    - 若输入smooth_scales，则
      $$
        input = x\cdot smooth_scales
      $$
      $$
        scale=row\_max(abs(input))/127
      $$
      $$
        y=round(input/scale)
      $$
  - 非对称量化：
    - 若不输入smooth_scales，则
      $$
        scale=(row\_max(x) - row\_min(x))/scale\_opt
      $$
      $$
        offset=offset\_opt-row\_max(x)/scale
      $$
      $$
        y=round(x/scale+offset)
      $$
    - 若输入smooth_scales，则
      $$
        input = x\cdot smooth_scales
      $$
      $$
        scale=(row\_max(input) - row\_min(input))/scale\_opt
      $$
      $$
        offset=offset\_opt-row\_max(input)/scale
      $$
      $$
        y=round(input/scale+offset)
      $$
  
  其中：
  - row_max代表每行求最大值。
  - row_min代表每行求最小值。
  - 当输出y类型为INT8时，scale_opt为255.0，offset_opt为127.0。
  - 当输出y类型为INT4时，scale_opt为15.0，offset_opt为7.0。

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
      <td><ul><li>算子输入的Tensor，对应公式中的`x`。</li><li>shape维度要大于1。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scales</td>
      <td>可选输入</td>
      <td><ul><li>算子输入的Tensor，对应公式中的`smooth_scales`。</li><li>数据类型与x的数据类型保持一致。当group_index为空时，形状为1维。Dim[0]是x的最后一个维度；当group_index不为空时，形状为2维。Dim[0]是专家数(E)。E必须不大于1024。Dim[1]是x的最后一个维度。</li></ul></td>
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
      <td><ul><li>量化后的输出张量，对应公式中的`y`。</li><li>shape与输入x的保持一致。当y的数据类型为INT4，x的最后一个维度必须能被2整除。</li></ul></td>
      <td>INT8、INT4、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输出</td>
      <td><ul><li>量化后的缩放系数，对应公式中的`scale`。</li><li>当quant_mode为"pertoken"时，形状与x去掉最后一个维度后的形状相同。当quant_mode为"pertensor"时，形状为(1,)。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输出</td>
      <td><ul><li>非对称量化使用的offset，对应公式中的`offset`。</li><li>shape与scale的shape保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term> ：
  - 输入`x`：数据类型仅支持FLOAT16。
  - 输出`y`：数据类型仅支持INT8。
  - 输入`smooth_scales`、`group_index`为预留参数，当前版本不参与计算。
  - 输入`dst_type`：只支持配置为2。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：输出`y`的数据类型仅支持INT8、INT4。

## 约束说明

- E不应大于x去掉最后一个维度后的维度的乘积结果(S)。
- `group_index`的值应递增，范围从0到S。最后一个值应为S。否则结果将无意义。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_dynamic_quant_v2](examples/test_aclnn_dynamic_quant_v2.cpp) | 通过[aclnnDynamicQuantV2](docs/aclnnDynamicQuantV2.md)接口方式调用DynamicQuantV2算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/dynamic_quant_v2_proto.h)构图方式调用DynamicQuantV2算子。         |