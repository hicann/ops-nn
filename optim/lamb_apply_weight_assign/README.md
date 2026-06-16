# LambApplyWeightAssign

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

- 算子功能：完成LAMB优化器的信任比(trust ratio)权重更新：依据权重范数input0与梯度范数input1计算信任比，再用学习率input2缩放update后原地更新参数input_param。

- 计算公式：

  $ratio = where(input0>0,\ where(input1>0,\ input0/input1,\ 1),\ 1)$

  $input\_param = input\_param - input2 \times ratio \times input3$

## 参数说明

<table><thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>数据类型</th><th>数据格式</th></tr>
</thead><tbody>
    <tr><td>input0</td><td>输入</td><td>公式中的input0。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input1</td><td>输入</td><td>公式中的input1。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input2</td><td>输入</td><td>公式中的input2。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input3</td><td>输入</td><td>公式中的input3。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input_param</td><td>输入</td><td>公式中的input_param。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input_param</td><td>输出</td><td>公式中的input_param。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
</tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_lamb_apply_weight_assign](./examples/test_geir_lamb_apply_weight_assign.cpp) | 通过算子IR构图方式调用LambApplyWeightAssign算子。 |
