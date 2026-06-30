# SparseApplyFtrl

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：SparseApplyFtrl是FTRL-Proximal（Follow The Regularized Leader）优化器的稀疏更新算子。根据`indices`选择`var`、`accum`、`linear`的对应行，使用梯度`grad`进行FTRL-Proximal更新。该算子广泛用于推荐系统、CTR预估等场景中的稀疏特征权重更新。

- 计算公式：

$$
\text{accum\_new} = \text{accum} + \text{grad}^2
$$

$$
\sigma = \frac{\text{accum}^{-\text{lr\_power}} - \text{accum\_new}^{-\text{lr\_power}}}{\text{lr}}
$$

$$
\text{linear\_new} = \text{linear} + \text{grad} + \sigma \cdot \text{var}
$$

$$
x = \text{clamp}(\text{linear\_new}, -l1, l1) - \text{linear\_new}
$$

$$
y = \frac{\text{accum\_new}^{-\text{lr\_power}}}{\text{lr}} + 2 \cdot l2
$$

$$
\text{var\_new} = \frac{x}{y}
$$

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
      <td>var</td>
      <td>输入</td>
      <td>待更新的变量tensor，≥2维</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输入</td>
      <td>梯度平方累加器，shape与var相同</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>linear</td>
      <td>输入</td>
      <td>线性累加器，shape与var相同</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>梯度tensor，shape[1:] == var.shape[1:]</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>索引向量，1-D，shape[0] == grad.shape[0]</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>输入</td>
      <td>学习率，标量tensor</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l1</td>
      <td>输入</td>
      <td>L1正则化系数，标量tensor</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l2</td>
      <td>输入</td>
      <td>L2正则化系数，标量tensor</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr_power</td>
      <td>输入</td>
      <td>学习率幂次，标量tensor</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输出</td>
      <td>更新后的变量，shape/dtype同输入var</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输出</td>
      <td>更新后的累加器，shape/dtype同输入accum</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>linear</td>
      <td>输出</td>
      <td>更新后的线性累加器，shape/dtype同输入linear</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>属性</td>
      <td>是否使用锁保护更新操作（当前实现未使用）</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

1. var/accum/linear必须具有相同的shape和dtype（float32）。
2. grad.shape[1:]必须等于var.shape[1:]，grad.shape[0]必须等于indices.shape[0]。
3. indices必须为1-D tensor。
4. lr/l1/l2/lr_power为标量tensor（0-D）。
5. 重复indices的结果不可预测（不做去重处理）。
6. indices越界时行为不可预期（不做运行时检查）。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a>-</a></td>
    <td>通过[算子IR]<a href="op_graph/sparse_apply_ftrl_proto.h">(..op_graph/sparse_apply_ftrl_proto.h)</a>构图方式调用SparseApplyFtrl算子。</td>
  </tr>
</tbody>
</table>
