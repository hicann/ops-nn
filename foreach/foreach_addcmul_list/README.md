# ForeachAddcmulList

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：先对张量列表x2和张量列表x3执行逐元素乘法，并将结果乘以张量scalars，最后将之前计算的结果与张量列表x1执行逐元素相加。
- 计算公式：

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}], x3 = [{x3_0}, {x3_1}, ... {x3_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = x1_{i} + {\rm scalars} × x2_{i} × x3_{i} (i=0,1,...n-1)
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
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>支持空Tensor。表示混合运算中加法的第一个输入张量列表，对应公式中的`x1`。该参数中所有Tensor的数据类型保持一致，每个Tensor的维度数（Dim）不大于8。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>支持空Tensor。表示混合运算中乘法的第二个输入张量列表，对应公式中的`x2`。数据类型和数据格式与入参`x1`一致，shape size（元素总数）与`x1`对应位置的Tensor一致；每个Tensor的维度数（Dim）不大于8。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>输入</td>
      <td>支持空Tensor。表示混合运算中乘法的第三个输入张量列表，对应公式中的`x3`。数据类型和数据格式与入参`x1`一致，shape size（元素总数）与`x1`对应位置的Tensor一致；每个Tensor的维度数（Dim）不大于8。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalars</td>
      <td>输入</td>
      <td>不支持空Tensor。表示混合运算中乘法的第一个输入张量，对应公式中的`scalars`。元素个数与`x1`中Tensor的个数相等。数据类型和数据格式与入参`x1`的数据类型和数据格式一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>支持空Tensor。表示混合运算的输出张量列表，对应公式中的`y`。数据类型和数据格式与入参`x1`的数据类型和数据格式一致，shape size大于等于入参`x1`的shape size。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030处理器系列产品：不支持BFLOAT16。
- Ascend 950PR/Ascend 950DT：scalars的数据类型与入参`x1`一致（`x1`为FLOAT32/FLOAT16/BFLOAT16/INT32时，scalars为对应的相同类型）。

## 约束说明

- 张量列表`x1`、`x2`、`x3`包含的Tensor个数必须相等，且与`y`及`scalars`的元素个数一致，单个Tensor列表中Tensor的数量不超过256个。
- 张量列表`x1`、`x2`、`x3`对应位置的Tensor的shape size（元素总数）必须相等。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_foreach_addcmul_list](examples/test_aclnn_foreach_addcmul_list.cpp) | 通过[aclnnForeachAddcmulList](docs/aclnnForeachAddcmulList.md)接口方式调用ForeachAddcmulList算子。 |
