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

  $y1 = input\_mul4 \times mul4\_x + \frac{y2/input\_realdiv0}{\sqrt{y3/input\_realdiv1} + add2\_y}$

  $y4 = \frac{y2/input\_realdiv0}{\sqrt{y3/input\_realdiv1} + add2\_y}$

## 参数说明

<table><thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>数据类型</th><th>数据格式</th></tr>
</thead><tbody>
    <tr><td>input_mul3</td><td>输入</td><td>公式中的input_mul3。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input_mul2</td><td>输入</td><td>公式中的input_mul2。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input_realdiv1</td><td>输入</td><td>公式中的input_realdiv1。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input_mul1</td><td>输入</td><td>公式中的input_mul1。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input_mul0</td><td>输入</td><td>公式中的input_mul0。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input_realdiv0</td><td>输入</td><td>公式中的input_realdiv0。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input_mul4</td><td>输入</td><td>公式中的input_mul4。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>mul0_x</td><td>输入</td><td>公式中的mul0_x。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>mul1_sub</td><td>输入</td><td>公式中的mul1_sub。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>mul2_x</td><td>输入</td><td>公式中的mul2_x。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>mul3_sub1</td><td>输入</td><td>公式中的mul3_sub1。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>mul4_x</td><td>输入</td><td>公式中的mul4_x。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>add2_y</td><td>输入</td><td>公式中的add2_y。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>y1</td><td>输出</td><td>公式中的y1。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>y2</td><td>输出</td><td>公式中的y2。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>y3</td><td>输出</td><td>公式中的y3。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>y4</td><td>输出</td><td>公式中的y4。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
</tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_lamb_next_m_v](./examples/test_geir_lamb_next_m_v.cpp) | 通过算子IR构图方式调用LambNextMV算子。 |
