# ForeachLog2

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

- 算子功能：对张量列表中的每一个元素执行以2为底的对数函数运算。
- 计算公式：

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = {\rm log}_2(x_i) (i=0,1,...n-1)
  $$

- 输入值域与特殊值语义：输入应为正数。对浮点输入元素`x`：
  - `x = 0`：输出负无穷（-inf）。
  - `x < 0`：输出NaN。
  - `x = +inf`：输出正无穷（+inf）。
  - `x = -inf`：输出NaN。
  - `x = NaN`：输出NaN（透传）。
  - `x > 0`（正常正数）：输出log2(x)。

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
      <td>表示按元素以2为底做对数函数运算的输入张量列表，对应公式中的`x`。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示按元素以2为底做对数函数运算的输出张量列表，对应公式中的`y`。数据类型和数据格式与入参`x`的数据类型和数据格式一致，shape size等于入参`x`的shape size。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：不支持BFLOAT16。

## 约束说明

- 输出不支持非连续Tensor。
- x与out的Tensor数量一致，且对应位置Tensor的shape完全相等。
- x或out的单个Tensor列表内所有Tensor的数据类型保持一致，且x与out的数据类型一致，均须在参数表声明的支持范围内。
- 单个Tensor的维度不超过8。
- x/y中单个Tensor列表的Tensor数量不能超过256。
- x与out数量为0（空列表）时直接返回成功，不执行计算。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_log2](examples/test_aclnn_foreach_log2.cpp) | 通过[aclnnForeachLog2](docs/aclnnForeachLog2.md)接口方式调用ForeachLog2算子。 |
| GE图模式 | -  | 通过[算子IR](op_graph/foreach_log2_proto.h)构图方式调用ForeachLog2算子。         |
