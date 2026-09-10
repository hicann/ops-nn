# LambApplyOptimizerAssign

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | × |
|  <term>Kirin 9030 处理器系列产品</term> | × |

## 功能说明

- 算子功能：对模型中的一个参数完成LAMB优化器的Adam矩更新与偏差校正后的update计算（含可选权重衰减），并原地更新一阶矩inputm与二阶矩inputv。

- 计算公式：

  $next\_v = inputv \times mul2\_x + grad^2 \times mul3\_x$

  $next\_m = inputm \times mul0\_x + grad \times mul1\_x$

  $output0 = \frac{next\_m / (1 - mul0\_x^{steps})}{\sqrt{next\_v / (1 - mul2\_x^{steps})} + add2\_y} + input3 \times weight\_decay\_rate \times do\_use\_weight$

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
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的grad（梯度）。<b>不参与广播</b>：shape必须与inputv、inputm完全相同。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputv</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的inputv（二阶矩）。<b>原地(in-place)更新，其shape即为输出shape，不参与广播</b>；须与grad、inputm完全相同。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputm</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的inputm（一阶矩）。<b>原地(in-place)更新，其shape即为输出shape，不参与广播</b>；须与grad、inputv完全相同。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input3</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input3（参与权重衰减的参数）。<b>唯一参与广播的输入</b>：按右对齐broadcast规则向inputv对齐，维度数可少于inputv（不可多于），对应维需相等或为1。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul0_x</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的mul0_x（beta1），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul1_x</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的mul1_x（1-beta1），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul2_x</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的mul2_x（beta2），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul3_x</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的mul3_x（1-beta2），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>add2_y</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的add2_y（epsilon），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>steps</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的steps（步数），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>do_use_weight</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的do_use_weight（是否使用权重衰减），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_decay_rate</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的weight_decay_rate（权重衰减率），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output0</td>
      <td>输出</td>
      <td>支持空Tensor。公式中的output0（update），shape与inputv、inputm相同。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputv</td>
      <td>输出</td>
      <td>支持空Tensor。更新后的inputv（二阶矩，原地更新），shape与输入inputv相同。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputm</td>
      <td>输出</td>
      <td>支持空Tensor。更新后的inputm（一阶矩，原地更新），shape与输入inputm相同。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- shape约束：`grad`、`inputv`、`inputm`三者shape必须**完全相同**，并直接决定三个输出的shape。`inputv`与`inputm`是原地更新的动量输出，不会被广播放大；`grad`同样不参与广播。**广播只发生在`input3`上**：`input3`按右对齐broadcast规则向`inputv`对齐，维度数可少于`inputv`（含标量），对应维需与`inputv`相等或为1；`input3`的维度数大于`inputv`、或某一维大于`inputv`对应维，均不被支持。

- `mul0_x`、`mul1_x`、`mul2_x`、`mul3_x`、`add2_y`、`steps`、`do_use_weight`、`weight_decay_rate`这8个标量输入不支持空Tensor；`grad`、`inputv`、`inputm`、`input3`支持空Tensor。

- 所有输入的数据类型必须一致，同为FLOAT16或同为FLOAT。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_lamb_apply_optimizer_assign](./examples/test_geir_lamb_apply_optimizer_assign.cpp) | 通过算子IR构图方式调用LambApplyOptimizerAssign算子。 |
