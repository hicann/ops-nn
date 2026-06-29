# SparseApplyFtrlV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- **算子功能**：对指定索引的稀疏梯度执行Ftrl-proximal V2优化算法更新，含`l2_shrinkage`正则项，用于大规模稀疏特征训练场景。

- **计算公式**：

对`indices`中每个索引`i`：

$$
\begin{aligned}
\text{grad\_shrink} &= \begin{cases} \text{grad}[i] & l2\_shrinkage = 0 \\ \text{grad}[i] + 2 \times l2\_shrinkage \times \text{var}[i] & \text{otherwise} \end{cases} \\
\text{accum}_{\text{new}}[i] &= \text{accum}[i] + \text{grad}[i]^2 \\
\text{accum\_pow}[i] &= \text{accum}[i]^{-lr\_power} \\
\text{accum\_new\_pow}[i] &= \text{accum}_{\text{new}}[i]^{-lr\_power} \\
\text{delta} &= (\text{accum\_pow}[i] - \text{accum\_new\_pow}[i]) / lr \\
\text{linear}_{\text{new}}[i] &= \text{linear}[i] + \text{grad\_shrink} + \text{delta} \times \text{var}[i] \\
\text{l1\_adjust} &= \text{clamp}(\text{linear}_{\text{new}}[i], -l1, l1) \\
x &= \text{l1\_adjust} - \text{linear}_{\text{new}}[i] \\
y &= \text{accum\_new\_pow}[i] / lr + 2 \times l2 \\
\text{var}_{\text{new}}[i] &= x / y
\end{aligned}
$$

> 通常`lr_power = -0.5`。仅`indices`中指定的行被更新，其余行保持不变。`indices[i]`越界（<0 或 >=N）则跳过该行。

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
      <td>var</td>
      <td>输入 / 输出</td>
      <td>待优化参数，shape [N, D1, D2, ...]</td>
      <td>float32, float16, bfloat16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输入 / 输出</td>
      <td>梯度平方累计，shape同var</td>
      <td>同var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>linear</td>
      <td>输入 / 输出</td>
      <td>线性项累计，shape同var</td>
      <td>同var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>稀疏梯度，shape [M, D1, D2, ...]</td>
      <td>同var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>稀疏索引，1-D向量，shape [M]</td>
      <td>int32, int64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>输入</td>
      <td>学习率，标量</td>
      <td>同var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l1</td>
      <td>输入</td>
      <td>L1正则系数，标量</td>
      <td>同var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l2</td>
      <td>输入</td>
      <td>L2正则系数，标量</td>
      <td>同var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l2_shrinkage</td>
      <td>输入</td>
      <td>L2收缩系数，标量</td>
      <td>同var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr_power</td>
      <td>输入</td>
      <td>学习率幂次，标量</td>
      <td>同var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>属性</td>
      <td>是否使用锁机制（多线程安全），默认false</td>
      <td>bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- var、accum、linear的shape必须完全一致。
- var至少为1维（N >= 1）。
- indices必须为1维向量。
- grad.dim(0) 必须等于indices.dim(0)（即M = len(indices)）。
- var.shape[1:] 必须等于grad.shape[1:]，即尾维度匹配。
- lr、l1、l2、l2_shrinkage、lr_power必须为标量（shape为[] 或 [1]）。
- indices中的值应在[0, var.shape[0]) 范围内且应唯一。若indices含重复值，结果不可预测（与TensorFlow行为一致）。
- 所有数据输入（var/accum/linear/grad/lr/l1/l2/l2_shrinkage/lr_power）的dtype必须一致。
- 空indices（len(indices) == 0）时跳过计算，var/accum/linear保持不变。
- lr应大于0，否则分母为0产生Inf。
- fp16/bf16内部以fp32计算后cast回原类型。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式 | [test_geir_sparse_apply_ftrl_v2](./examples/arch35/test_geir_sparse_apply_ftrl_v2.cpp) | 通过[算子IR](./op_graph/sparse_apply_ftrl_v2_proto.h)构图方式调用SparseApplyFtrlV2算子。 |
