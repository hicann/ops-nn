# InplaceApplyAdaMax

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

- **算子功能**：执行AdaMax优化器的单步参数更新。AdaMax是Adam优化器的变体，使用无穷范数 $L_{\infty}$ 代替二阶矩估计，对权重`var`、一阶矩`m`、无穷范数`v`进行更新。输出端口名 `var`/`m`/`v` 与输入同名（GE inplace 别名），框架将输出内存别名到输入内存，实现原地更新。

- **计算公式**：

  给定时间步 $t$ 的梯度 $g_t$，衰减系数 $\beta_1, \beta_2$，学习率 $lr$，数值稳定常数 $\epsilon$，以及外部传入的偏差校正因子 $\beta_1^t$：

  $$
  \begin{aligned}
  m_{t} &= \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \\
  v_{t} &= \max(\beta_2 \cdot v_{t-1},\ |g_t|) \\
  var_{t} &= var_{t-1} - \frac{lr}{1 - \beta_1^t} \cdot \frac{m_t}{v_t + \epsilon}
  \end{aligned}
  $$

  算子原型：9输入 + 3输出 (var/m/v，输出名与输入同名 = GE inplace 别名) + 1属性 (use_locking)。对齐 canndev `REG_OP(ApplyAdaMaxD)`。

## 参数说明

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 600px">
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
    <td>输入</td>
    <td>待更新的权重参数张量，对应公式中的var。1-8维ND格式。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>m</td>
    <td>输入</td>
    <td>一阶矩张量，对应公式中的m。shape/dtype必须与var一致。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>v</td>
    <td>输入</td>
    <td>无穷范数估计张量（$L_{\infty}$），对应公式中的v。shape/dtype必须与var一致。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta1_power</td>
    <td>输入</td>
    <td>偏差校正因子 $\beta_1^t$，由外部传入。shape必须为 [1] 的scalar Tensor。须保证 $1 - \beta_1^t \neq 0$。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率，对应公式中的lr。shape必须为 [1] 的scalar Tensor。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta1</td>
    <td>输入</td>
    <td>一阶矩衰减系数 $\beta_1$，取值范围 [0, 1)。shape必须为 [1] 的scalar Tensor。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta2</td>
    <td>输入</td>
    <td>无穷范数衰减系数 $\beta_2$，取值范围 [0, 1)。shape必须为 [1] 的scalar Tensor。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>输入</td>
    <td>数值稳定常数 $\epsilon$，必须大于0。shape必须为 [1] 的scalar Tensor。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>当前梯度 $g_t$，对应公式中的grad。shape/dtype必须与var一致。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>属性</td>
    <td>语义占位的bool属性，默认false。当前实现不强制互斥锁。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var</td>
    <td>输出</td>
    <td>更新后的权重张量，与输入var共享Device内存（inplace别名）。shape/dtype与输入var完全相同。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>m</td>
    <td>输出</td>
    <td>更新后的一阶矩张量，与输入m共享Device内存（inplace别名）。shape/dtype与输入m完全相同。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>v</td>
    <td>输出</td>
    <td>更新后的无穷范数张量，与输入v共享Device内存（inplace别名）。shape/dtype与输入v完全相同。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|--------|------|
| 图模式 | [test_geir_inplace_apply_ada_max](examples/arch35/test_geir_inplace_apply_ada_max.cpp) | 通过[算子IR](op_graph/inplace_apply_ada_max_proto.h)构图方式调用InplaceApplyAdaMax算子。 |
