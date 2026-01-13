# Modulate

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：实现特征的自适应缩放（scale）和平移（shift）
- 计算公式：

  $$
  y = x \times (1 + scaleOptional) + shiftOptional
  $$

  其中 $self \in R ^ {B \times L \times D}$，$scaleOptional \in R ^ {B \times 1 \times D}$，$shiftOptional \in R ^ {B \times 1 \times D}$（或者 $scaleOptional \in R ^ {B \times D}$，$shiftOptional \in R ^ {B \times D}$）

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>表示传入的特征张量，公式中的x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>输入</td>
      <td>可选参数，表示缩放的系数，公式中的scaleOptional。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shiftOptional</td>
      <td>输入</td>
      <td>可选参数，表示平移的系数，公式中的shiftOptional。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示缩放后的输出张量，公式中的y。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_modulate](./examples/test_aclnn_modulate.cpp) | 通过[aclnnModulate](./docs/aclnnModulate.md)接口方式调用Modulate算子。    |


