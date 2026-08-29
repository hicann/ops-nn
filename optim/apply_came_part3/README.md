# ApplyCamePart3

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

- **算子功能**：计算CAME（Confidence-guided Adaptive Memory Efficient）优化器第三阶段的一阶矩更新值，以及行、列和全局归约结果。

- **计算公式**：

  对二维输入张量 `u` 和 `m`，以及标量输入 `eps`、`beta1`、`clip_threshold`、`sum_square_u`，设全局行数为 `global_n`、全局列数为 `global_m`，先计算：

  $$
  s = \max\left(1,\frac{sum\_square\_u}{global\_n \times global\_m \times clip\_threshold}\right)
  $$

  实现遵循A2的条件分支语义：仅当缩放值大于1时使用该值，其余情况使用1。

  $$
  m\_update_{i,j} = (1 - beta1) \times \frac{u_{i,j}}{s} + beta1 \times m_{i,j}
  $$

  `use_first_moment`为`true`时，输出`m`为`m_update`；为`false`时，输出`m`保留输入`m`。

  再令：

  $$
  x_{i,j}=\left(\frac{u_{i,j}}{s}-m\_update_{i,j}\right)^2+eps
  $$

  分别进行行、列和全局归约：

  $$
  \begin{aligned}
  sum\_u\_r_i &= \sum_j x_{i,j} \\
  sum\_u\_c_j &= \sum_i x_{i,j} \\
  sum\_u\_rc &= \sum_i \sum_j x_{i,j}
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
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>u</td>
    <td>输入</td>
    <td>二维输入张量，公式中的<code>u</code>。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>m</td>
    <td>输入</td>
    <td>一阶矩输入张量，形状必须与<code>u</code>相同，公式中的<code>m</code>。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>eps</td>
    <td>输入</td>
    <td>数值稳定项，标量或单元素一维张量。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta1</td>
    <td>输入</td>
    <td>一阶矩衰减系数，标量或单元素一维张量。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>clip_threshold</td>
    <td>输入</td>
    <td>裁剪阈值，标量或单元素一维张量。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sum_square_u</td>
    <td>输入</td>
    <td><code>u</code>平方和或对应的全局统计值，标量或单元素一维张量。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>global_shape</td>
    <td>可选输入</td>
    <td>全局二维形状，包含全局行数和列数。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_first_moment</td>
    <td>属性</td>
    <td>是否输出更新后的一阶矩，默认为<code>false</code>。</td>
    <td>Bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>m</td>
    <td>输出</td>
    <td>一阶矩输出张量，形状和数据类型与输入<code>m</code>相同。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sum_u_r</td>
    <td>输出</td>
    <td>按列归约后的行结果，形状为<code>[N]</code>。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sum_u_c</td>
    <td>输出</td>
    <td>按行归约后的列结果，形状为<code>[M]</code>。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sum_u_rc</td>
    <td>输出</td>
    <td>全局归约结果，形状为<code>[1]</code>。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- `u`和输入`m`必须为形状相同的二维ND张量，两个维度都必须大于0，且每个维度不超过`INT32_MAX`。
- `eps`、`beta1`、`clip_threshold`和`sum_square_u`必须为FLOAT32类型的标量或单元素一维张量。
- `global_shape`为可选INT64类型输入，必须是一维长度为2的张量`[global_n, global_m]`；未提供时使用输入`u`的二维形状进行归约计算。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_apply_came_part3.cpp](./examples/test_geir_apply_came_part3.cpp) | 通过[算子IR](./op_graph/apply_came_part3_proto.h)构图方式调用ApplyCamePart3算子。 |
