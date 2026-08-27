# InplaceApplyFtrlV2

<!-- codespell:ignore FTRL Ftrl -->

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

InplaceApplyFtrlV2是FTRL-Proximal（Follow The Regularized Leader - Proximal）在线学习优化算法的单步参数更新算子，用于推荐系统与CTR预估等大规模稀疏特征训练阶段。算子根据当前batch梯度`grad`与一组超参，原地更新模型参数`var`、梯度平方累积`accumulation`、线性项`linear`三个跨调用持久的状态张量，并显式输出更新后的三者；通过L1正则化将`|linear| ≤ threshold`的权重直接置零，产生稀疏解。

记 `gs = grad + 2.0 * l2_shrinkage * var`，`accumulation_new = accumulation + grad * grad`，`Δpow = accumulation_new^(-lr_power) - accumulation^(-lr_power)`。更新公式为：

$$
linear_{new} = linear + gs - \frac{\Delta pow \cdot var}{lr}
$$

$$
var_{new} = \begin{cases} \frac{l1 \cdot sign(linear_{new}) - linear_{new}}{accumulation_{new}^{(-lr\_power)}/lr + 2.0 \cdot l2}, & |linear_{new}| > l1 \\ 0, & |linear_{new}| \le l1 \end{cases}
$$

`l2_shrinkage=0`时`gs=grad`，退化为FTRL V1行为。`out_accumulation = accumulation_new`（使用原始`grad`，非`gs`）。

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
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
</tr>
</thead>
<tbody>
<tr>
<td>var</td>
<td>输入</td>
<td>待更新的模型参数，对应公式中var；与输出var同名，是否复用buffer由图编译器决定。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>accum</td>
<td>输入</td>
<td>梯度平方累积（FTRL自适应学习率状态，跨调用持久），对应公式中accum。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>linear</td>
<td>输入</td>
<td>线性项（FTRL状态变量，跨调用持久），对应公式中linear。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>grad</td>
<td>输入</td>
<td>当前batch梯度（只读），对应公式中grad。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>lr</td>
<td>输入</td>
<td>学习率（标量），对应公式中lr；按numpy规则广播到var.shape。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>l1</td>
<td>输入</td>
<td>L1正则化系数（标量），控制稀疏性，对应公式中l1。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>l2</td>
<td>输入</td>
<td>L2正则化系数（标量），影响分母y，对应公式中l2。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>l2_shrinkage</td>
<td>输入</td>
<td>L2收缩系数（标量），对应公式中l2_shrinkage；为0时退化为V1行为。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>lr_power</td>
<td>输入</td>
<td>lr幂次（标量，通常-0.5），对应公式中lr_power。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>var</td>
<td>输出</td>
<td>更新后的权重，对应公式中var；与输入var同名，允许图编译器复用输入buffer，但算子不强制物理地址相同。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>accum</td>
<td>输出</td>
<td>更新后的累积，对应公式中accum；与输入accum同名，允许图编译器复用输入buffer，但算子不强制物理地址相同。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>linear</td>
<td>输出</td>
<td>更新后的线性项，对应公式中linear；与输入linear同名，允许图编译器复用输入buffer，但算子不强制物理地址相同。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>use_locking</td>
<td>可选属性</td>
<td>是否使用互斥锁，默认false；NPU无Ref语义，仅支持false。</td>
<td>-</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- 仅支持ND数据格式。
- 数据类型约束：
  - 支持BFLOAT16、FLOAT16、FLOAT三种数据类型。
  - 全部9路输入（var/accumulation/linear/grad + lr/l1/l2/l2_shrinkage/lr_power）与3路输出的数据类型必须完全相同（以var为准，不支持混合精度）。
  - 幂运算等中间计算在FLOAT32下进行后回降为输入类型。
  - 不支持FLOAT64及整型（INT8/INT16/INT32/INT64等）。
- Shape约束：
  - var/accumulation/linear/grad四路张量的shape必须完全一致，不支持张量间广播。
  - 四路张量的rank范围为[0, 8]，仅拒绝rank大于8的输入。
  - lr/l1/l2/l2_shrinkage/lr_power五路标量输入为rank-0，按numpy规则广播到var.shape。
  - 支持空Tensor（shape含0维，如[0, N]），短路返回空输出。
  - 支持动态Shape（-1未知维度）与动态Rank（-2未知秩）。
- 属性约束：
  - use_locking：仅支持false（NPU无Ref语义）。
- 仅在Ascend 950PR/Ascend 950DT上注册，其他产品不支持。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:---------|:---------|:-----|
| 图模式 | [test_geir_inplace_apply_ftrl_v2_dynamic](./examples/test_geir_inplace_apply_ftrl_v2_dynamic.cpp) | 通过 [算子IR](./op_graph/inplace_apply_ftrl_v2_proto.h) 构图方式调用 InplaceApplyFtrlV2 算子，覆盖动态 shape（-1/-2）场景。 |
