# ForeachMinimumScalarList

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 算子功能：对张量列表x和标量列表scalars执行逐元素比较，计算每个元素对应的最小值。

- 计算公式：

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  scalars = [{scalars_0}, {scalars_1}, ... {scalars_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = \min(x_i, \text{scalars}_i) (i=0,1,...n-1)
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
      <td>表示取最小值运算的输入张量列表，对应公式中的`x`。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalars</td>
      <td>输入</td>
      <td>表示取最小值运算的输入标量列表，对应公式中的`scalars`。元素个数与`x`中Tensor的个数相等。数据类型与输入参数的数据类型具有一定对应关系：当入参`x`的数据类型为FLOAT32、FLOAT16、BFLOAT16时，`scalars`的数据类型支持FLOAT32。<!--、DOUBLE-->；当入参`x`的数据类型为INT32时，`scalars`的数据类型仅支持INT64。</td>
      <td>FLOAT32、INT64</td><!--aclnn多了一个double-->
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示取最小值运算的输出张量列表，对应公式中的`y`。数据类型和数据格式与入参`x`的数据类型和数据格式一致，shape size大于等于入参`x`的shape size。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_foreach_minimum_scalar_list](examples/test_aclnn_foreach_minimum_scalar_list.cpp) | 通过[aclnnForeachMinimumScalarList](docs/aclnnForeachMinimumScalarList.md)接口方式调用ForeachMinimumScalarList算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/foreach_minimum_scalar_list_proto.h)构图方式调用ForeachMinimumScalarList算子。         |