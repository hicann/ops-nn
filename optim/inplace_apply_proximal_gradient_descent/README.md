# InplaceApplyProximalGradientDescent

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

InplaceApplyProximalGradientDescent对稠密变量执行一次带L1、L2正则的逐元素近端梯度更新。本算子仅提供GE图模式接口，并通过独立输出`var_out`返回更新结果。

对`var`和`delta`中的每个同位置元素，先计算：

$$
prox\_v_i = var_i - alpha \times delta_i
$$

再根据共享标量`l1`选择分支：

$$
var\_out_i =
\begin{cases}
\dfrac{\operatorname{sign}(prox\_v_i) \times
\max\left(\left|prox\_v_i\right| - alpha \times l1, 0\right)}
{1 + alpha \times l2}, & l1 > 0, \\
\dfrac{prox\_v_i}{1 + alpha \times l2}, & l1 \le 0.
\end{cases}
$$

其中$\operatorname{sign}(0)=0$，`alpha`、`l1`和`l2`由全部元素共享。

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
<td>待更新的变量张量，是输出shape和数据类型的推导源。</td>
<td>BF16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>alpha</td>
<td>输入</td>
<td>全部元素共享的学习率，shape仅支持0-D或[1]。</td>
<td>BF16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>l1</td>
<td>输入</td>
<td>全部元素共享的L1参数，shape仅支持0-D或[1]；其值决定计算分支。</td>
<td>BF16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>l2</td>
<td>输入</td>
<td>全部元素共享的L2参数，shape仅支持0-D或[1]。</td>
<td>BF16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>delta</td>
<td>输入</td>
<td>参与逐元素更新的稠密增量，shape必须与var逐维相同。</td>
<td>BF16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>var_out</td>
<td>输出</td>
<td>更新后的变量张量，shape和数据类型均与var相同。</td>
<td>BF16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>use_locking</td>
<td>可选属性</td>
<td>资源锁定兼容属性，不改变计算公式，默认值为false。</td>
<td>BOOL</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- `var`与`delta`的rank范围为0到16，二者必须逐维完全同形，不支持广播。
- `alpha`、`l1`、`l2`各自仅支持0-D或`[1]`。
- 五个输入必须采用相同的数据类型，仅支持全BF16、全FLOAT16或全FLOAT三种组合。
- 所有输入和输出仅支持ND数据格式。
- `var_out`是独立输出，不承诺与`var`复用物理地址。
- `alpha`、`l1`和`l2`不限制正负值；`1 + alpha * l2`为零或负数时仍执行浮点除法。

## 调用说明

|调用方式|样例代码|说明|
|:---|:---|:---|
|GE图模式|[test_geir_inplace_apply_proximal_gradient_descent.cpp](examples/arch35/test_geir_inplace_apply_proximal_gradient_descent.cpp)|通过[算子IR](op_graph/inplace_apply_proximal_gradient_descent_proto.h)构图，并校验输出shape、数据类型和数值。|
