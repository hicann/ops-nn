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

- 算子功能：计算CAME（Confidence-guided Adaptive Memory Efficient）优化器第一阶段的平方梯度归约值。

- 计算公式：

  对输入`grad`最后两维$[N, M]$上的元素$g_{b,\dots,i,j}$，先计算：

  $$
  x_{b,\dots,i,j} = g_{b,\dots,i,j}^{2} + eps
  $$

  再对每个批次切片分别进行行、列和二维归约：

  $$
  \begin{aligned}
  sum\_grad\_r_{b,\dots,i} &= \sum_j x_{b,\dots,i,j} \\
  sum\_grad\_c_{b,\dots,j} &= \sum_i x_{b,\dots,i,j} \\
  sum\_grad\_rc_{b,\dots} &= \sum_i \sum_j x_{b,\dots,i,j}
  \end{aligned}
  $$

  其中，$b,\dots$表示批次维度索引，$i$和$j$分别表示最后两维的索引。

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
      <td>梯度张量，最后两维为<code>[N, M]</code>，具体shape要求见约束说明。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>输入</td>
      <td>数值稳定性常数，具体shape要求见约束说明。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_grad_r</td>
      <td>输出</td>
      <td>对最后一维进行归约得到的结果，具体shape见约束说明。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  <tr>
    <td>sum_grad_c</td>
    <td>输出</td>
    <td>对倒数第二维进行归约得到的结果，具体shape见约束说明。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
    <tr>
      <td>sum_grad_rc</td>
      <td>输出</td>
      <td>对最后两维进行归约得到的结果，具体shape见约束说明。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- `grad`必须为非空ND张量，所有维度的大小均必须大于0。
- `eps`必须为仅含一个FLOAT类型元素的ND张量。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - `grad`的rank不小于2，最后两维为`[N, M]`，其余维度为批次维度。
  - `grad`所有维度大小的乘积不能超过INT64的最大值。
  - `eps`支持标量或shape为`[1]`的一维张量。
  - `sum_grad_r`、`sum_grad_c`和`sum_grad_rc`的shape分别为`[batch..., N]`、`[batch..., M]`和`[batch...]`；当`grad`的rank为2时，`sum_grad_rc`为标量。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - `grad`仅支持shape为`[N, M]`的二维张量。
  - `eps`仅支持标量。
  - `sum_grad_r`、`sum_grad_c`和`sum_grad_rc`的shape分别为`[N]`、`[M]`和`[1]`。
- 本算子仅支持GE图模式调用，不提供公开的aclnn接口。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_apply_came_part1.cpp](./examples/test_geir_apply_came_part1.cpp) | 通过[算子IR](./op_graph/apply_came_part1_proto.h)构图方式调用ApplyCamePart1算子。 |
