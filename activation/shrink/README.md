# Shrink

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- **算子功能**：对输入张量进行非线性收缩变换，根据输入值与阈值`lambd`的关系将元素映射到三个区域（正区域、负区域、死区），常用于稀疏化模型推理和训练中的权重正则化。

- **计算公式**：

  $$
  out_i=\begin{cases}
  self_i-bias, & self_i > lambd \\
  self_i+bias, & self_i < -lambd \\
  0, & -lambd \leq self_i \leq lambd
  \end{cases}
  $$

  其中`lambd`为阈值参数（默认0.5），`bias`为偏移量参数（默认0.0）。当`lambd < 0`时，内部钳位为0。

## 参数说明

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 300px">
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
  </tr></thead>
<tbody>
  <tr>
    <td>input_x</td>
    <td>输入</td>
    <td>输入张量，对应公式中的self。元素值与lambd比较决定归属区域。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lambd</td>
    <td>属性</td>
    <td>阈值参数，对应公式中的lambd。当self_i>lambd时进入正区域，self_i&lt;-lambd时进入负区域。默认值为0.5。当lambd&lt;0时，内部钳位为0。</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>属性</td>
    <td>偏移量参数，对应公式中的bias。在正区域减去bias，在负区域加上bias。默认值为0.0。</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>output_y</td>
    <td>输出</td>
    <td>输出张量，对应公式中的out。与input_x的dtype和shape完全相同。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

 无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_shrink](./examples/test_aclnn_shrink.cpp) | 通过[aclnnShrink](./docs/aclnnShrink.md)接口方式调用Shrink算子。 |
| 图模式 | - | 通过[算子IR](./op_graph/shrink_proto.h)构图方式调用Shrink算子。 |
