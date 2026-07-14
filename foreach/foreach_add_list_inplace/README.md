# ForeachAddListInplace

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

- 算子功能：对两个张量列表执行逐元素x1+alpha*x2，结果原地写回第一个列表。
- 计算公式：

  $$
  x1_i = x1_i + alpha \times x2_i
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
      <td>x1</td>
      <td>输入/输出</td>
      <td>支持空Tensor。表示加法运算的第一个输入张量列表，对应公式中的`x1`，同时作为输出，计算结果原地写回该参数。该参数中所有Tensor的数据类型保持一致，每个Tensor的维度数不超过8维。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>支持空Tensor。表示加法运算的第二个输入张量列表，对应公式中的`x2`。数据类型、数据格式和shape与入参`x1`的数据类型、数据格式和shape一致。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>不支持空Tensor。表示加法运算中第二个输入的系数，对应公式中的`alpha`。元素个数为1。数据类型与入参`x1`的数据类型具有一定对应关系：当`x1`的数据类型为FLOAT32、FLOAT16、INT32时，数据类型与`x1`的数据类型保持一致；当`x1`的数据类型为BFLOAT16时，数据类型支持FLOAT32。</td>
      <td>FLOAT32、FLOAT16、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 入参`x1`与`x2`中Tensor的数量必须相同，且单个Tensor列表包含的Tensor数量不超过256个。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn接口 | [test_aclnn_foreach_add_list_inplace](examples/arch35/test_aclnn_foreach_add_list_inplace.cpp) | 通过[aclnnForeachAddListInplace](docs/aclnnForeachAddListInplace.md)接口方式调用ForeachAddListInplace算子。 |
