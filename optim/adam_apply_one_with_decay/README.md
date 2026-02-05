 # AdamApplyOneWithDecay

##  产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：对模型中的一个参数（如权重），完成Adam优化算法的单步计算和更新。

- 计算公式：

$output_0 = input_0^2 \times mul_3\_x + input_1 \times mul_2\_x$

$output_1 = input_2 \times mul_0\_x + input_0 \times mul_1\_x$

$output_2 = input_3 - ((\frac{output_1}{\sqrt(output_0) + add_2\_y}) + input_3 \times mul_4\_x) \times input_4$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
<col style="width: 100px">
<col style="width: 150px">
<col style="width: 280px">
<col style="width: 330px">
<col style="width: 120px">
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
    <td>input0</td>
    <td>输入</td>
    <td>待进行adam_apply_one_with_decay计算的入参，公式中的input0。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>input1</td>
    <td>输入</td>
    <td>待进行adam_apply_one_with_decay计算的入参，公式中的input1。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>input2</td>
    <td>输入</td>
    <td>待进行adam_apply_one_with_decay计算的入参，公式中的input2。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>input3</td>
    <td>输入</td>
    <td>待进行adam_apply_one_with_decay计算的入参，公式中的input3。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>input4</td>
    <td>输入</td>
    <td>待进行adam_apply_one_with_decay计算的入参，公式中的input4。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mul0_x</td>
    <td>输入</td>
    <td>待进行adam_apply_one_with_decay计算的入参，公式中的mul0_x。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mul1_x</td>
    <td>输入</td>
    <td>待进行adam_apply_one_with_decay计算的入参，公式中的mul1_x。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mul2_x</td>
    <td>输入</td>
    <td>待进行adam_apply_one_with_decay计算的入参，公式中的mul2_x。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mul3_x</td>
    <td>输入</td>
    <td>待进行adam_apply_one_with_decay计算的入参，公式中的mul3_x。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mul4_x</td>
    <td>输入</td>
    <td>待进行adam_apply_one_with_decay计算的入参，公式中的mul4_x。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>add2_y</td>
    <td>输入</td>
    <td>待进行adam_apply_one_with_decay计算的入参，公式中的add2_y。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>output0</td>
    <td>输出</td>
    <td>待进行adam_apply_one_with_decay计算的出参，公式中的output0。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>output1</td>
    <td>输出</td>
    <td>待进行adam_apply_one_with_decay计算的出参，公式中的output1。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>output2</td>
    <td>输出</td>
    <td>待进行adam_apply_one_with_decay计算的出参，公式中的output2。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_adam_apply_one_with_decay](./examples/test_geir_adam_apply_one_with_decay.cpp)   | 通过[算子IR](./op_graph/adam_apply_one_with_decay_proto.h)构图方式调用AdamApplyOne算子。 |