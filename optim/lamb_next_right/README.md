# LambNextRight

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | × |
|  <term>Kirin 9030 处理器系列产品</term> | × |

## 功能说明

- 算子功能：BERT LAMB优化器图融合算子：计算二阶矩next_v及其偏差校正后的分母sqrt(v_unbiased)+epsilon。

- 计算公式：

  $y1 = input\_mul2 \times mul2\_x + input\_square^2 \times mul3\_x\quad(next\_v)$

  $y2 = \sqrt{y1 \times truediv1\_recip} + add2\_y$

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
      <td>input_square</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_square（梯度g），主张量，shape需与input_mul2满足broadcast关系。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul2</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_mul2（二阶矩v），主张量，shape需与input_square满足broadcast关系，其broadcast结果决定各输出的shape。</td>
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
      <td>truediv1_recip</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的truediv1_recip（偏差校正分母的倒数），标量。</td>
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
      <td>y1</td>
      <td>输出</td>
      <td>支持空Tensor。公式中的y1（next_v），shape取input_square与input_mul2的broadcast结果。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>支持空Tensor。公式中的y2（偏差校正分母），shape取input_square与input_mul2的broadcast结果。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 所有输入的数据类型必须一致，同为FLOAT16或同为FLOAT。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_lamb_next_right](./examples/test_geir_lamb_next_right.cpp) | 通过算子IR构图方式调用LambNextRight算子。 |
