# QuantMax

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |


## 功能说明

- 算子功能：根据输入的scale对输入x进行量化，并计算输入x的绝对值的最大值amax。

- 计算公式：

  $$
  y = \text{Cast}(x \times scale, \text{dst\_type}, \text{round\_mode})
  $$

  $$
  amax = \max(|x|)
  $$

  其中，$x$表示输入张量，$scale$表示量化缩放因子，$\text{dst\_type}$指定输出类型，$\text{round\_mode}$指定舍入模式，$amax$为输入张量所有元素绝对值的最大值。


## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>待量化的输入张量，对应公式中的x。支持1-8维任意shape。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>量化缩放因子，对应公式中的scale。shape固定为[1]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>量化后的输出张量，对应公式中的y。shape与x相同，数据类型由dst_type决定。</td>
      <td>HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>amax</td>
      <td>输出</td>
      <td>x绝对值的最大值，对应公式中的amax。shape固定为[1]，数据类型与x一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>属性</td>
      <td>取值需为Cast API支持的有效舍入模式。</td>
      <td>string</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>属性</td>
      <td>指定输出量化数据类型。34=HIFLOAT8，35=FLOAT8_E5M2，36=FLOAT8_E4M3。</td>
      <td>int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- `round_mode`与`dst_dtype`的匹配关系：
    - 当`dst_dtype`为`35`或`36`时，`round_mode`必须为`"rint"`。
    - 当`dst_dtype`为`34`时，`round_mode`必须为`"round"`或`"hybrid"`。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_quant_max](./examples/arch35/test_aclnn_quant_max.cpp) | 通过[aclnnQuantMax](./docs/aclnnQuantMax.md)接口方式调用QuantMax算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/quant_max_proto.h)构图方式调用QuantMax算子。 |
