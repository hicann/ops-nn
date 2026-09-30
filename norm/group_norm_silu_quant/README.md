# GroupNormSiluQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：计算输入self的组归一化，输出均值meanOut，标准差的倒数rstdOut，以及对silu的输出结果进行量化的结果out。

- 计算公式：
  - **GroupNorm：**

    记 $E[x] = \bar{x}$代表$x$的均值，$Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$代表$x$的方差，则

    $$
    \left\{
    \begin{array} {rcl}
    groupNormOut& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
    meanOut& &= E[x]\\
    rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
    \end{array}
    \right.
    $$

  - **Silu：**

    $$
    siluOut = \frac{groupNormOut}{1+e^{-groupNormOut}}
    $$

  - **Quant：**

    $$
    out = round(siluOut / quantScale)
    $$

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
      <td>表示待归一化的输入张量，对应公式中的`x`。维度为2~8，第0维为N、第1维为C。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示归一化后的缩放张量，对应公式中的$\gamma$。数据类型与`x`一致，shape为`[C]`（C为`x`的第1维）。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>表示归一化后的偏移张量，对应公式中的$\beta$。数据类型与`x`一致，shape为`[C]`（C为`x`的第1维）。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quantScale</td>
      <td>输入</td>
      <td>表示量化缩放系数，对应公式中的`quantScale`。shape为`[1]`（per-tensor）或`[C]`（per-channel，C为`x`的第1维）。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_groups</td>
      <td>属性</td>
      <td>表示将`x`的第1维（C）分为`num_groups`组，需能整除C。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>表示归一化时加在方差上的扰动量，对应公式中的$\epsilon$，默认值为1e-05。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_silu</td>
      <td>属性</td>
      <td>表示是否对归一化结果做Silu激活，默认值为true；取false时跳过激活直接量化。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>表示量化后的输出张量，对应公式中的`yOut`。shape与`x`一致。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>meanOut</td>
      <td>输出</td>
      <td>表示每组的均值，对应公式中的`meanOut`。数据类型与`x`一致，shape为(N, num_groups)。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstdOut</td>
      <td>输出</td>
      <td>表示每组标准差的倒数，对应公式中的`rstdOut`。数据类型与`x`一致，shape为(N, num_groups)。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `x`的维度为2~8，第0维为N、第1维为C；`num_groups`需能整除C。
- `gamma`、`beta`的shape需为`[C]`，数据类型与`x`一致。
- `quantScale`的shape需为`[1]`（per-tensor）或`[C]`（per-channel）。
- 本节描述算子（图模式）的约束。aclnn接口另有其自身约束，参见[aclnnGroupNormSiluQuant](docs/aclnnGroupNormSiluQuant.md)。
- 支持空Tensor：N（第0维）与C（第1维）需大于0，其余维度可为0；此时`yOut`为空，`meanOut`填充为0、`rstdOut`填充为NAN。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_group_norm_silu_quant](examples/arch22/test_aclnn_group_norm_silu_quant.cpp) | 通过[aclnnGroupNormSiluQuant](docs/aclnnGroupNormSiluQuant.md)接口方式调用GroupNormSiluQuant算子。 |
| 图模式调用 | [test_geir_group_norm_silu_quant](examples/arch35/test_geir_group_norm_silu_quant.cpp) | 通过[算子IR](op_graph/group_norm_silu_quant_proto.h)构图方式调用GroupNormSiluQuant算子。 |
