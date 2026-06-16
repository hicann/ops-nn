# LambUpdateWithLrV2

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

- 算子功能：BERT LAMB优化器图融合算子（信任比权重更新，无裁剪）：以阈值greater_y与回退值select_e计算信任比，再用学习率x3更新参数x5。

- 计算公式：

  $ratio = where(x1>greater\_y,\ where(x2>greater\_y,\ x1/x2,\ select\_e),\ select\_e)$

  $y = x5 - x3 \times ratio \times x4$

## 参数说明

<table><thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>数据类型</th><th>数据格式</th></tr>
</thead><tbody>
    <tr><td>x1</td><td>输入</td><td>公式中的x1。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>x2</td><td>输入</td><td>公式中的x2。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>x3</td><td>输入</td><td>公式中的x3。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>x4</td><td>输入</td><td>公式中的x4。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>x5</td><td>输入</td><td>公式中的x5。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>greater_y</td><td>输入</td><td>公式中的greater_y。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>select_e</td><td>输入</td><td>公式中的select_e。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>y</td><td>输出</td><td>公式中的y。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
</tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_lamb_update_with_lr_v2](./examples/test_geir_lamb_update_with_lr_v2.cpp) | 通过算子IR构图方式调用LambUpdateWithLrV2算子。 |
