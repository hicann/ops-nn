# InplaceApplyFtrl

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

InplaceApplyFtrl 实现 FTRL-Proximal（Follow The Regularized Leader）优化器的单步参数更新，用于稀疏特征场景下的在线学习与大规模推荐模型训练。算子读取当前权重 `var`、梯度平方累积器 `accum`、线性累加器 `linear` 与当前步梯度 `grad`，按 FTRL 公式就地更新 `var`、`accum`、`linear` 三个状态量；`lr`、`l1`、`l2`、`lr_power` 为标量超参。内部计算固定使用 fp32，输入与输出可为 bfloat16、float16 或 float32。

标准 V1 更新公式为：

$$
\begin{aligned}
accum_{new} &= accum + grad \cdot grad \\
\sigma &= |accum_{new}|^{-lr\_power} - |accum|^{-lr\_power} \\
linear_{new} &= linear + grad - \frac{\sigma}{lr} \cdot var \\
y &= \frac{|accum_{new}|^{-lr\_power}}{lr} + 2 \cdot l2 \\
x &= l1 \cdot sign(linear_{new}) - linear_{new} \\
var &= \begin{cases} x / y, & |linear_{new}| > l1 \\ 0, & |linear_{new}| \le l1 \end{cases} \\
accum &= accum_{new},\quad linear = linear_{new}
\end{aligned}
$$

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
<td>模型权重变量 w，就地更新的状态量，shape 与 accum、linear、grad 完全一致。</td>
<td>bfloat16、float16、float32</td>
<td>ND</td>
</tr>
<tr>
<td>accum</td>
<td>输入</td>
<td>梯度平方累积器 n（n ≥ 0），就地更新的状态量。</td>
<td>bfloat16、float16、float32</td>
<td>ND</td>
</tr>
<tr>
<td>linear</td>
<td>输入</td>
<td>线性累加器 z，就地更新的状态量。</td>
<td>bfloat16、float16、float32</td>
<td>ND</td>
</tr>
<tr>
<td>grad</td>
<td>输入</td>
<td>当前步梯度 g，与 var 同 shape 同 dtype。</td>
<td>bfloat16、float16、float32</td>
<td>ND</td>
</tr>
<tr>
<td>lr</td>
<td>输入</td>
<td>学习率 η（η > 0），rank-0 标量，运行时广播到 var 的 shape。</td>
<td>bfloat16、float16、float32</td>
<td>ND</td>
</tr>
<tr>
<td>l1</td>
<td>输入</td>
<td>L1 正则化强度 λ₁（λ₁ ≥ 0），rank-0 标量。</td>
<td>bfloat16、float16、float32</td>
<td>ND</td>
</tr>
<tr>
<td>l2</td>
<td>输入</td>
<td>L2 正则化强度 λ₂（λ₂ ≥ 0），rank-0 标量。</td>
<td>bfloat16、float16、float32</td>
<td>ND</td>
</tr>
<tr>
<td>lr_power</td>
<td>输入</td>
<td>学习率衰减幂次 p（p ≤ 0，通常取 -0.5），rank-0 标量。</td>
<td>bfloat16、float16、float32</td>
<td>ND</td>
</tr>
<tr>
<td>var</td>
<td>输出</td>
<td>更新后的权重 w'，与输入 var 同 shape 同 dtype，就地写回 var。</td>
<td>bfloat16、float16、float32</td>
<td>ND</td>
</tr>
<tr>
<td>accum</td>
<td>输出</td>
<td>更新后的梯度平方累加器 n'，与输入 accum 同 shape 同 dtype，就地写回 accum。</td>
<td>bfloat16、float16、float32</td>
<td>ND</td>
</tr>
<tr>
<td>linear</td>
<td>输出</td>
<td>更新后的线性累加器 z'，与输入 linear 同 shape 同 dtype，就地写回 linear。</td>
<td>bfloat16、float16、float32</td>
<td>ND</td>
</tr>
<tr>
<td>use_locking</td>
<td>可选属性</td>
<td>是否对更新加锁，默认 false，当前仅支持 false。</td>
<td>Bool</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- `var`、`accum`、`linear`、`grad` 四个张量的 shape 必须完全相同，不支持张量间广播。
- `lr`、`l1`、`l2`、`lr_power` 必须为 rank-0 标量（0-d Tensor），数据类型与 `var` 一致。
- 八个输入端口必须使用相同的数据类型，不支持混合 dtype，仅支持 bfloat16、float16、float32 三种组合。
- 张量秩 rank ∈ [0, 8]。
- 数据格式仅支持 ND。
- 内部计算固定使用 fp32，bfloat16 与 float16 输入会先提升为 fp32 计算再回写原 dtype。
- `use_locking` 当前仅支持默认值 false。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | [test_geir_inplace_apply_ftrl_dynamic.cpp](/examples/test_geir_inplace_apply_ftrl_dynamic.cpp) | 通过 GE 图模式调用，算子 IR 定义见 [算子IR](op_graph/inplace_apply_ftrl_proto.h)。 |
