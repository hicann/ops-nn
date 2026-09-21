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
      <td>支持空Tensor。公式中的grad（梯度），shape支持0-8维，需与其他输入满足broadcast规则。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputv</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的inputv（二阶矩）。<b>原地(in-place)更新，其shape即为输出shape</b>；须与inputm完全相同，且全部输入的broadcast结果必须恰好等于该shape。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputm</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的inputm（一阶矩）。<b>原地(in-place)更新，其shape即为输出shape</b>；须与inputv完全相同，且全部输入的broadcast结果必须恰好等于该shape。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input3</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input3（参与权重衰减的参数），shape支持0-8维，需与其他输入满足broadcast规则。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul0_x</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的mul0_x（beta1），shape支持0-8维，需与其他输入满足broadcast规则。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul1_x</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的mul1_x（1-beta1），shape支持0-8维，需与其他输入满足broadcast规则。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul2_x</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的mul2_x（beta2），shape支持0-8维，需与其他输入满足broadcast规则。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul3_x</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的mul3_x（1-beta2），shape支持0-8维，需与其他输入满足broadcast规则。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>add2_y</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的add2_y（epsilon），shape支持0-8维，需与其他输入满足broadcast规则。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>steps</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的steps（步数），shape支持0-8维，需与其他输入满足broadcast规则。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>do_use_weight</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的do_use_weight（是否使用权重衰减），shape支持0-8维，需与其他输入满足broadcast规则。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_decay_rate</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的weight_decay_rate（权重衰减率），shape支持0-8维，需与其他输入满足broadcast规则。</td>
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

- shape约束：所有输入的shape需两两满足broadcast规则。`inputv`与`inputm`是原地(in-place)更新的动量输出，两者shape必须**完全相同**，且全部输入的broadcast结果必须恰好等于该shape（否则原地写回会越界）；三个输出的shape均取该shape。其余输入（含`grad`）可向其广播。
- 所有输入及输出的维度数为0~8。当全部输入均为0维时，三个输出的shape为(1,)。


- 所有输入的数据类型必须一致，同为FLOAT16或同为FLOAT。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_lamb_apply_optimizer_assign](./examples/test_geir_lamb_apply_optimizer_assign.cpp) | 通过算子IR构图方式调用LambApplyOptimizerAssign算子。 |
