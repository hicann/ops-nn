# ForeachSubScalarInplace

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

- 算子功能：对输入张量列表中的每个张量逐元素减去同一个标量（原地）。
- 计算公式：

  $$
  x_i = x_i - scalar
  $$

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
      <td>x</td>
      <td>输入/输出</td>
      <td>支持空Tensor。表示减法运算的输入张量列表，对应公式中的`x`，同时作为输出，计算结果原地写回该参数。该参数中所有Tensor的数据类型保持一致，每个Tensor的维度数（Dim）不大于8。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalar</td>
      <td>输入</td>
      <td>不支持空Tensor。表示减法运算的输入张量，对应公式中的`scalar`。元素个数为1，维度数（Dim）不大于8。数据类型与入参`x`的数据类型具有一定对应关系：当`x`的数据类型为FLOAT32、INT32时，数据类型与`x`的数据类型保持一致；当`x`的数据类型为BFLOAT16时，数据类型支持FLOAT32；当`x`的数据类型为FLOAT16时，数据类型支持FLOAT16、FLOAT32。</td>
      <td>FLOAT32、FLOAT16、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 入参`x`单个Tensor列表包含的Tensor数量不超过256个。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn接口 | [test_aclnn_foreach_sub_scalar_inplace](examples/arch35/test_aclnn_foreach_sub_scalar_inplace.cpp) | 通过[aclnnForeachSubScalarInplace](docs/aclnnForeachSubScalarInplace.md)接口方式调用ForeachSubScalarInplace算子。 |
