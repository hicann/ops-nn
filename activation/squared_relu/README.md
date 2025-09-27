# SquaredRelu

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：

  SquaredRelu函数是一个基于标准Relu函数的变体，其主要特点是对Relu函数的输出进行平方，常作为模型的激活函数。

- 计算公式：

  $$
  y_i=(Relu(x_i))^2
  $$

  其中Relu的计算公式：

  $$
  Relu(x_i)=max(0,x_i)
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 773px"><colgroup>
  <col style="width: 91px">
  <col style="width: 154px">
  <col style="width: 156px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>公式中的输入x。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入值为nan, 输出也为nan, 输入是inf, 输出也是inf。
- 输入是-inf, 输出是0。
- 输入shape只支持维度1至8维，不在范围内执行报错。 

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_squared_relu](./examples/test_aclnn_squared_relu.cpp) | 通过[aclnnSquaredRelu](./docs/aclnnSquaredRelu.md)接口方式调用SquaredRelu算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/squared_relu_proto.h)构图方式调用SquaredRelu算子。 |