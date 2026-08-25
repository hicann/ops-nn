# ApplyCamePart1

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term> | √ |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
|  <term>Atlas 200I/500 A2 推理产品</term> | × |
|  <term>Atlas 推理系列产品</term> | × |
|  <term>Atlas 训练系列产品</term> | × |

## 功能说明

- **算子功能**：计算CAME（Confidence-guided Adaptive Memory Efficient）优化器第一阶段的平方梯度归约值。

- **计算公式**：

  对二维输入张量 `grad` 中的元素 $g_{i,j}$，以及标量张量 `eps`，先计算：

  $$
  x_{i,j} = g_{i,j}^{2} + eps
  $$

  再分别进行行、列和全局归约：

  $$
  \begin{aligned}
  sum\_grad\_r_i &= \sum_j x_{i,j} \\
  sum\_grad\_c_j &= \sum_i x_{i,j} \\
  sum\_grad\_rc &= \sum_i \sum_j x_{i,j}
  \end{aligned}
  $$

## 参数说明

<table style="table-layout: fixed; width: 100%">
<colgroup>
<col style="width: 16%">
<col style="width: 18%">
<col style="width: 38%">
<col style="width: 16%">
<col style="width: 12%">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>梯度张量，公式中的<code>grad</code>，rank不小于2；最后两维为<code>[N, M]</code>，前导维为batch维。</td>
    <td>FLOAT16、FLOAT32、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>eps</td>
    <td>输入</td>
    <td>数值稳定项，公式中的<code>eps</code>，为标量或单元素一维张量。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sum_grad_r</td>
    <td>输出</td>
    <td>按列归约后的行结果，shape为<code>[batch..., N]</code>，对应公式中的<code>sum_grad_r</code>。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sum_grad_c</td>
    <td>输出</td>
    <td>按行归约后的列结果，shape为<code>[batch..., M]</code>，对应公式中的<code>sum_grad_c</code>。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sum_grad_rc</td>
    <td>输出</td>
    <td>全局归约结果，shape为<code>[batch...]</code>，对应公式中的<code>sum_grad_rc</code>。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- `grad` rank不小于2，数据格式为ND，所有维度都必须大于0；最后两维为<code>[N, M]</code>，前导维作为batch维。
- `eps` 为FLOAT32类型、ND格式的标量或单元素一维张量。
- `sum_grad_r`、`sum_grad_c`和`sum_grad_rc`的数据类型均为FLOAT32。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_apply_came_part1.cpp](./examples/test_geir_apply_came_part1.cpp) | 通过[算子IR](./op_graph/apply_came_part1_proto.h)构图方式调用ApplyCamePart1算子。 |
