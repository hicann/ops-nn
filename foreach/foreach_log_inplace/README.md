# ForeachLogInplace

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

- 算子功能：对输入张量列表中的每个张量执行逐元素自然对数运算（原地）。
- 计算公式：

  $$
  x_i = \ln(x_i)
  $$

## 参数说明

<table><thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>数据类型</th><th>数据格式</th></tr>
</thead><tbody>
    <tr><td>x</td><td>输入/输出</td><td>待计算的张量列表（原地更新）</td><td>FLOAT16、FLOAT、BFLOAT16</td><td>ND</td></tr>
</tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn接口 | [test_aclnn_foreach_log_inplace](examples/arch35/test_aclnn_foreach_log_inplace.cpp) | 通过[aclnnForeachLogInplace](docs/aclnnForeachLogInplace.md)接口方式调用ForeachLogInplace算子。 |
