# ForeachMulListInplace

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

- 算子功能：对两个张量列表执行逐元素相乘，结果原地写回第一个列表。
- 计算公式：

  $$
  x1_i = x1_i \times x2_i
  $$

## 参数说明

<table><thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>数据类型</th><th>数据格式</th></tr>
</thead><tbody>
    <tr><td>x1</td><td>输入/输出</td><td>第一个张量列表（原地更新）</td><td>FLOAT16、FLOAT、BFLOAT16、INT32</td><td>ND</td></tr>
    <tr><td>x2</td><td>输入</td><td>第二个张量列表</td><td>FLOAT16、FLOAT、BFLOAT16、INT32</td><td>ND</td></tr>
</tbody></table>

## 约束说明

- inplace操作的输入/输出不支持非连续Tensor。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn接口 | [test_aclnn_foreach_mul_list_inplace](examples/arch35/test_aclnn_foreach_mul_list_inplace.cpp) | 通过[aclnnForeachMulListInplace](docs/aclnnForeachMulListInplace.md)接口方式调用ForeachMulListInplace算子。 |
