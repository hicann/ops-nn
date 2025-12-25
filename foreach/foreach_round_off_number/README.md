# ForeachRoundOffNumber

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：对输入张量列表的每个张量执行指定精度的四舍五入运算，可通过roundMode指定舍入方式。
- 计算公式：

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$ 

  $$
  y_i = round(x_i, roundMode) (i=0,1,...n-1)
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
      <td>表示进行舍入运算的输入张量列表，对应公式中的`x`。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>roundMode</td>
      <td>输入</td>
      <td>表示进行舍入计算的输入张量，对应公式中的`roundMode`。元素个数为1。roundMode的取值及对应的舍入策略如下：当roundMode=1，表示对输入进行四舍六入五成双舍入操作；当roundMode=2，表示对输入进行向负无穷舍入取整操作；当roundMode=3，表示对输入进行向正无穷舍入取整操作；当roundMode=4，表示对输入进行四舍五入舍入操作；当roundMode=5，表示对输入进行向零舍入操作；当roundMode=6，表示对输入进行最近邻奇数舍入操作；当roundMode为其他时，如果精度损失会进行四舍六入五成双舍入操作，不涉及精度损失时则不进行舍入操作。</td>
      <td>INT8、INT64</td><!--V2多了INT64，原型原来也没有INT64-->
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示进行舍入运算的输出张量列表，对应公式中的`y`。该参数中所有Tensor的数据类型保持一致。数据类型和数据格式与入参`x`的数据类型和数据格式一致，shape size大于等于入参`x`的shape size。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_foreach_round_off_number](examples/test_aclnn_foreach_round_off_number.cpp) | 通过[aclnnForeachRoundOffNumber](docs/aclnnForeachRoundOffNumber.md)接口方式调用ForeachRoundOffNumber算子。 |
| aclnn接口  | [test_aclnn_foreach_round_off_number_v2](examples/test_aclnn_foreach_round_off_number_v2.cpp) | 通过[aclnnForeachRoundOffNumberV2](docs/aclnnForeachRoundOffNumberV2.md)接口方式调用ForeachRoundOffNumberV2算子。 |
| 图模式 | - | 通过[算子IR](op_graph/foreach_round_off_number_proto.h)构图方式调用ForeachRoundOffNumber算子。         |