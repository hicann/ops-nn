# LambNextMV

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

- 算子功能：BERT LAMB优化器图融合算子：基于上游已算好的中间量（g^2、一/二阶矩、偏差校正分母等），完成Adam矩更新与偏差校正后的update计算。

- 计算公式：

  $y3 = input\_mul2 \times mul2\_x + input\_mul3 \times mul3\_sub1\quad(next\_v)$

  $y2 = input\_mul0 \times mul0\_x + input\_mul1 \times mul1\_sub\quad(next\_m)$

  $y1 = input\_mul4 \times mul4\_x + \frac{y2/input\_realdiv0}{\sqrt{y3/input\_realdiv1 + add2\_y}}$

  $y4 = \frac{y2/input\_realdiv0}{\sqrt{y3/input\_realdiv1} + add2\_y}$

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
      <td>input_mul3</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_mul3（g^2），主张量，shape需与input_mul0满足broadcast关系。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul2</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_mul2（二阶矩v），主张量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_realdiv1</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_realdiv1（1-beta2^t），主张量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul1</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_mul1（梯度g），主张量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul0</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_mul0（一阶矩m），主张量，shape需与input_mul3满足broadcast关系，其broadcast结果决定各输出的shape。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_realdiv0</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_realdiv0（1-beta1^t），主张量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mul4</td>
      <td>输入</td>
      <td>支持空Tensor。公式中的input_mul4（参数param），主张量。</td>
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
      <td>mul1_sub</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的mul1_sub（1-beta1），标量。</td>
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
      <td>mul3_sub1</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的mul3_sub1（1-beta2），标量。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mul4_x</td>
      <td>输入</td>
      <td>不支持空Tensor。公式中的mul4_x（权重衰减系数），标量。</td>
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
      <td>支持空Tensor。公式中的y1（update），shape取input_mul3与input_mul0的broadcast结果。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>支持空Tensor。公式中的y2（next_m），shape取input_mul3与input_mul0的broadcast结果。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y3</td>
      <td>输出</td>
      <td>支持空Tensor。公式中的y3（next_v），shape取input_mul3与input_mul0的broadcast结果。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y4</td>
      <td>输出</td>
      <td>支持空Tensor。公式中的y4，shape取input_mul3与input_mul0的broadcast结果。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 所有输入的数据类型必须一致，同为FLOAT16或同为FLOAT。
- input_mul0/input_mul1/input_mul2/input_mul3/input_mul4 为主张量，其shape需保持一致（或可相互广播到同一shape）；各输出y1/y2/y3/y4的shape均取该广播结果（实现以input_mul3与input_mul0的broadcast结果为准）。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_lamb_next_m_v](./examples/test_geir_lamb_next_m_v.cpp) | 通过算子IR构图方式调用LambNextMV算子。 |
