# LambUpdateWithLr

## 产品支持情况

|产品|是否支持|
|:---|:---:|
|<term>Ascend 950PR/Ascend 950DT</term>|√|

## 功能说明

- 算子功能：BERT LAMB优化器图融合算子（信任比权重更新，含裁剪）：计算信任比并用minimum_y/greater_y做上下界裁剪后更新参数input_sub。

- 计算公式：

$ratio = where(input\_greater1>greater\_y,\ input\_greater\_realdiv/input\_realdiv,\ select\_e)$

$clip = max(min(ratio,\ minimum\_y),\ greater\_y)$

$y = input\_sub - clip \times input\_mul0 \times input\_mul1$

## 参数说明

<table><thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>数据类型</th><th>数据格式</th></tr>
</thead><tbody>
    <tr><td>input_greater1</td><td>输入</td><td>公式中的input_greater1。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input_greater_realdiv</td><td>输入</td><td>公式中的input_greater_realdiv。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input_realdiv</td><td>输入</td><td>公式中的input_realdiv。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input_mul0</td><td>输入</td><td>公式中的input_mul0。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input_mul1</td><td>输入</td><td>公式中的input_mul1。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>input_sub</td><td>输入</td><td>公式中的input_sub。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>greater_y</td><td>输入</td><td>公式中的greater_y。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>select_e</td><td>输入</td><td>公式中的select_e。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>minimum_y</td><td>输入</td><td>公式中的minimum_y。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>y</td><td>输出</td><td>公式中的y。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
</tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_lamb_update_with_lr](./examples/test_geir_lamb_update_with_lr.cpp) | 通过算子IR构图方式调用LambUpdateWithLr算子。 |
