# ForeachLogInplace

## 产品支持情况

|产品|是否支持|
|:---|:---:|
|<term>Ascend 950PR/Ascend 950DT</term>|√|

## 功能说明

- 算子功能：对输入张量列表中的每个张量执行逐元素自然对数运算（原地）。
- 计算公式：

$x_i = \ln(x_i)$

## 参数说明

<table><thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>数据类型</th><th>数据格式</th></tr>
</thead><tbody>
    <tr><td>x</td><td>输入/输出</td><td>待计算的张量列表（原地更新）</td><td>FLOAT16、FLOAT、BFLOAT16</td><td>ND</td></tr>
</tbody></table>

## 约束说明

- inplace操作的输入/输出不支持非连续Tensor。
