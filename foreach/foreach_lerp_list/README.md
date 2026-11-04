# ForeachLerpList

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

- 算子功能：对两个张量列表对应位置元素执行插值计算，其中张量列表weight是插值系数。
- 计算公式：

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  weight = [{weight_0}, {weight_1}, ... {weight_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = x1_i + {\rm weight}_i × (x2_i - x1_i) (i=0,1,...n-1)
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
      <td>表示进行线性插值计算的第一个输入张量列表，对应公式中的<code>x1</code>。支持空Tensor。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示进行线性插值计算的第二个输入张量列表，对应公式中的<code>x2</code>。支持空Tensor。数据类型、数据格式和shape与入参<code>x1</code>的数据类型、数据格式和shape一致。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>表示进行线性插值计算的插值系数张量列表，对应公式中的<code>weight</code>。支持空Tensor。数据类型、数据格式和shape与入参<code>x1</code>的数据类型、数据格式和shape一致。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示进行线性插值计算的输出张量列表，对应公式中的<code>y</code>。支持空Tensor。数据类型和数据格式与入参<code>x1</code>的数据类型和数据格式一致，shape与入参<code>x1</code>一致，storage shape size不小于入参<code>x1</code>的storage shape size。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：不支持BFLOAT16。

## 约束说明

- 非连续Tensor：仅aclnn接口支持非连续输入与输出，接口内部先将非连续Tensor转为连续Tensor参与计算，输出结果再通过ViewCopy写回原输出视图；图模式（IR）与Kernel直调通路不支持非连续Tensor。
- x1、x2、weight、y 的维度范围为 0-8。
- out 的 shape 需与 x1 一致，且 storage shape size 不小于 x1。
- TensorList 长度上限为 256。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_foreach_lerp_list](examples/test_aclnn_foreach_lerp_list.cpp) | 通过[aclnnForeachLerpList](docs/aclnnForeachLerpList.md)接口方式调用ForeachLerpList算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/foreach_lerp_list_proto.h)构图方式调用ForeachLerpList算子。         |
