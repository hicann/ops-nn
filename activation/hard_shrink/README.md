# HardShrink

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     √    |
|  <term>Atlas推理系列产品</term>     |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：完成HardShrink激活函数计算，将输入张量中绝对值小于等于阈值lambd的元素置零，绝对值大于阈值的元素保持不变；输入为NaN的元素，输出为NaN。

- 计算公式：

$$
\text{HardShrink}(x) = \begin{cases} x, & \text{if } x > \lambda \\ x, & \text{if } x < -\lambda \\ 0, & \text{otherwise} \end{cases}
$$

其中，x为输入张量，$\lambda$为阈值参数lambd。

**特殊值说明**：输入为NaN时，输出为NaN。

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>input_x</td>
      <td>输入</td>
      <td>输入张量，对应公式中的x。支持0-8维，支持空Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambd</td>
      <td>属性</td>
      <td>阈值参数，对应公式中的λ，float类型标量；负值等效于0，即输入原样输出。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_y</td>
      <td>输出</td>
      <td>输出张量，对应公式中的HardShrink(x)。与input_x同shape、同dtype。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- input_x与output_y的数据类型必须一致，支持FLOAT、FLOAT16、BFLOAT16。
- input_x与output_y的shape必须一致，不涉及广播。
- input_x支持0-8维。
- input_x支持空Tensor（0元素），此时output_y也为空Tensor，不执行计算。
- lambd为float类型标量，取值无限制；负值等效于0，输入原样输出。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_hard_shrink](./examples/test_aclnn_hard_shrink.cpp) | 通过[aclnnHardshrink](./docs/aclnnHardshrink.md)接口方式调用HardShrink算子。    |
| GE图模式 | [hard_shrink_proto.h](./op_graph/hard_shrink_proto.h) | 通过`REG_OP(HardShrink)`图算子原型在GE图中调用HardShrink算子。    |
