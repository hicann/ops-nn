# ForeachNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 算子功能：对输入张量列表的每个张量进行范数运算。
- 计算公式：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    $$
    x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
    $$

    $$
    y = \left(\sum_{i=0}^{n-1}|x_i|^{p}\right)^{\frac{1}{{p}}}  (i=0,1,...n-1)
    $$

  - <term>Ascend 950PR/Ascend 950DT</term>：
    $$
    x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
    $$
    - scalar = +Inf:
      $$
      y = \max|x_i|
      $$
    - int(scalar) = 1:
      $$
      y = \sum_{i=0}^{n-1}|x_i|
      $$
    - other:
      $$
      y = \left(\sum_{i=0}^{n-1}|x_i|^{2}\right)^{\frac{1}{{2}}}
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
      <td>表示进行范数运算的输入张量列表，对应公式中的`x`。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalar</td>
      <td>输入</td>
      <td>表示进行范数运算的阶数，对应公式中的`p`。</td>
      <td>FLOAT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示进行范数运算的输出张量列表，对应公式中的`y`。数据类型、数据格式与入参`x`的数据类型、数据格式一致。shape size为1。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>


<term>Ascend 950PR/Ascend 950DT</term>：
- 输入`x`、输出`y`支持的最大长度为256个。
- 输入`scalar`的取值，仅支持1或者2。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_foreach_norm](examples/test_aclnn_foreach_norm.cpp) | 通过[aclnnForeachNorm](docs/aclnnForeachNorm.md)接口方式调用ForeachNorm算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/foreach_norm_proto.h)构图方式调用ForeachNorm算子。         |