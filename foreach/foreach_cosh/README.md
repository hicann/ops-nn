# ForeachCosh

## 产品支持情况

|产品|是否支持|
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

- 算子功能：按元素进行双曲余弦函数运算。
- 计算公式：

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = \cosh(x_i) (i=0,1,...n-1)
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
      <td>表示进行双曲余弦函数运算的输入张量列表，对应公式中的`x`。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示进行双曲余弦函数运算的输出张量列表，对应公式中的`y`。数据类型和数据格式与入参`x`的数据类型和数据格式一致，`x`与`y`的列表成员数量必须相等，逐成员shape需与入参`x`对应成员shape完全相等。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：不支持BFLOAT16。

## 约束说明

- 输出支持非连续Tensor，实现会先连续化输出，计算后回写到调用者的原始输出张量。
- 维度：`x`与`y`中每个Tensor支持0-8维；超过8维时在Tiling阶段返回失败。
- 数据格式：支持ND、NCHW、NHWC、HWCN、NDHWC、NCDHW；私有格式不支持。
- TensorList成员上限为256。
- 成员数量为0的空TensorList视为无操作：接口成功返回且不写入输出。

## 调用说明

|调用方式|样例代码|说明|
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_cosh](examples/test_aclnn_foreach_cosh.cpp) |通过[aclnnForeachCosh](docs/aclnnForeachCosh.md)接口方式调用ForeachCosh算子。|
| GE图模式| -  |通过[算子IR](op_graph/foreach_cosh_proto.h)构图方式调用ForeachCosh算子。|
