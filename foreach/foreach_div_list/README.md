# ForeachDivList

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

- 算子功能：对张量列表x1和张量列表x2执行逐元素除法。

- 计算公式：

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = \frac{x1_i}{x2_i} (i=0,1,...n-1)
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
      <td>表示除法运算的第一个输入张量列表，对应公式中的`x1`。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示除法运算的第二个输入张量列表，对应公式中的`x2`。数据类型、数据格式和shape与入参`x1`的数据类型、数据格式和shape一致，该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示除法运算的输出张量列表，对应公式中的`y`。数据类型和数据格式与入参`x1`的数据类型和数据格式一致，shape size大于等于入参`x1`的shape size。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：不支持BFLOAT16。

## 约束说明

- 张量列表`x1`、`x2`包含的Tensor个数必须相等，且与`y`的元素个数一致，单个Tensor列表中Tensor的数量不超过256个。
- `x1`、`x2`、`y`中每个Tensor的维度数不超过8且形状一致。
- `-1`（动态维度）与`-2`（动态Rank）标记由Host推导阶段按框架约定处理；内核侧Tiling要求shape已具体化（元素数非负）。
- 输出非连续Tensor支持情况：aclnnForeachDivList支持非连续输出Tensor（内部自动转连续并写回）；图模式（IR）不支持非连续输出Tensor。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_div_list](examples/test_aclnn_foreach_div_list.cpp) | 通过[aclnnForeachDivList](docs/aclnnForeachDivList.md)接口方式调用ForeachDivList算子。 |
| GE图模式 | -  | 通过[算子IR](op_graph/foreach_div_list_proto.h)构图方式调用ForeachDivList算子。         |
