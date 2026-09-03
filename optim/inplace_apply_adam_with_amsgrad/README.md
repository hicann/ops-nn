# InplaceApplyAdamWithAmsgrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     x    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

InplaceApplyAdamWithAmsgrad对AMSGrad优化器的四个状态张量var、m、v、vhat执行单步原地更新，语义对标TensorFlow ResourceApplyAdamWithAmsgrad。var、m、v、vhat在原地写回更新值；grad与六个标量超参参与计算，全部张量输入输出共享同一数据类型。该算子通过GE图模式调用。

计算公式如下，更新依赖顺序固定为m→v→vhat（使用新v）→var（使用新m）：

$$
\alpha = lr \times \frac{\sqrt{1 - beta2\_power}}{1 - beta1\_power}
$$

$$
m = m + (grad - m) \times (1 - beta1)
$$

$$
v = v + (grad \times grad - v) \times (1 - beta2)
$$

$$
vhat = where(vhat < v,\ v,\ vhat)
$$

$$
var = var - \frac{m \times \alpha}{\sqrt{vhat} + epsilon}
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
<td>待更新权重状态，原地写回，对应公式中var。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>m</td>
<td>输入</td>
<td>一阶矩状态，原地写回，对应公式中m。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>v</td>
<td>输入</td>
<td>二阶矩状态，原地写回，对应公式中v。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>vhat</td>
<td>输入</td>
<td>二阶矩running max状态，原地写回，对应公式中vhat。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>beta1_power</td>
<td>输入</td>
<td>标量超参β1^t，对应公式中beta1_power，由框架侧逐step维护。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>beta2_power</td>
<td>输入</td>
<td>标量超参β2^t，对应公式中beta2_power，由框架侧逐step维护。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>lr</td>
<td>输入</td>
<td>学习率标量超参，对应公式中lr。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>beta1</td>
<td>输入</td>
<td>标量超参β1，对应公式中beta1。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>beta2</td>
<td>输入</td>
<td>标量超参β2，对应公式中beta2。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>epsilon</td>
<td>输入</td>
<td>标量超参ε，对应公式中epsilon。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>grad</td>
<td>输入</td>
<td>梯度张量，对应公式中grad。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>use_locking</td>
<td>可选属性</td>
<td>语义占位属性，默认false，当前实现不执行加锁，数值行为等价于use_locking=false。</td>
<td>Bool</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- 11个张量输入必须共享同一数据类型（FLOAT、FLOAT16或BFLOAT16），混合dtype组合将被拒绝。
- var、m、v、vhat、grad五个状态张量shape必须完全一致，五者之间不支持广播。
- beta1_power、beta2_power、lr、beta1、beta2、epsilon必须为rank-0标量或shape为[1]的张量。
- 仅支持ND格式。
- 状态张量rank不超过8。

## 调用说明

<table style="table-layout: fixed; width: 1000px">
<colgroup>
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 600px">
</colgroup>
<thead>
<tr>
<th>调用方式</th>
<th>样例代码</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td>图模式调用</td>
<td><a href="./examples/test_geir_inplace_apply_adam_with_amsgrad.cpp">test_geir_inplace_apply_adam_with_amsgrad</a></td>
<td>通过算子IR构图方式调用InplaceApplyAdamWithAmsgrad算子。</td>
</tr>
</tbody>
</table>
