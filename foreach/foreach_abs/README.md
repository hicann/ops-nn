# ForeachAbs

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |
|  <term>Kirin X90处理器系列产品</term> | √ |
|  <term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- 算子功能：对输入张量列表中的每个张量执行逐元素绝对值运算。
- 计算公式：

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i=|{x_i}| (i=0,1,...n-1)
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
      <td>x</td>
      <td>输入</td>
      <td>表示进行绝对值计算的输入张量列表，对应公式中的`x`。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示进行绝对值计算的输出张量列表，对应公式中的`y`。该参数中所有Tensor的数据类型保持一致。数据类型与入参`x`的数据类型一致，shape与入参`x`对应位置的Tensor一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：不支持BFLOAT16。

## 约束说明

- x/y中单个Tensor列表的Tensor数量不能超过256。
- x与y的Tensor数量必须一致。
- x中所有Tensor的数据类型必须一致。
- y中对应位置的Tensor的数据类型与x一致。
- x/y中单个Tensor的维度取值范围为0-8维，超过时报错。
- y中对应位置的Tensor的shape必须与x一致。
- 支持空Tensor（元素数为0）。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_abs](examples/test_aclnn_foreach_abs.cpp) | 通过[aclnnForeachAbs](docs/aclnnForeachAbs.md)接口方式调用ForeachAbs算子。 |
| GE图模式 | -  | 通过[算子IR](op_graph/foreach_abs_proto.h)构图方式调用ForeachAbs算子。         |
