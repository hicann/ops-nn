# ForeachAddListInplace

## 产品支持情况

|产品|是否支持|
|:---|:---:|
|<term>Ascend 950PR/Ascend 950DT</term>|√|

## 功能说明

- 算子功能：对两个张量列表执行逐元素 x1+alpha*x2，结果原地写回第一个列表。
- 计算公式：

$x1_i = x1_i + alpha \times x2_i$

## 参数说明

<table><thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>数据类型</th><th>数据格式</th></tr>
</thead><tbody>
    <tr><td>x1</td><td>输入/输出</td><td>第一个张量列表（原地更新）</td><td>FLOAT16、FLOAT、BFLOAT16、INT32</td><td>ND</td></tr>
    <tr><td>x2</td><td>输入</td><td>第二个张量列表</td><td>FLOAT16、FLOAT、BFLOAT16、INT32</td><td>ND</td></tr>
    <tr><td>alpha</td><td>输入</td><td>标量系数</td><td>FLOAT16、FLOAT、BFLOAT16、INT32</td><td>ND</td></tr>
</tbody></table>

## 约束说明

无
