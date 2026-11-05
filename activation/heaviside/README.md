# Heaviside

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |
|  <term>kirin X90处理器系列产品</term>    |     √    |
|  <term>Kirin 9030处理器系列产品</term>    |     √    |

## 功能说明

- 接口功能：
  计算输入input中每个元素的Heaviside阶跃函数，作为模型的激活函数。

- 计算公式：

  $$
  \text{Heaviside}(\text{input},   \text{values}) =
  \begin{cases}
  0, & \text{如果input} < 0 \\
  \text{values}, & \text{如果input} = 0 \\
  1, & \text{如果input} > 0
  \end{cases}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 813px"><colgroup>
  <col style="width: 103px">
  <col style="width: 144px">
  <col style="width: 194px">
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
      <td>input</td>
      <td>输入</td>
      <td>公式中的输入input。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>输入</td>
      <td>公式中的输入values。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的Heaviside。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：不支持BFLOAT16。

## 约束说明

- input、values、out的数据类型必须一致。
- input支持1~8维；当values为单元素张量（标量）时，input额外支持0维（标量）。
- values仅支持以下两种形态：
  - 单元素张量（元素总数为1，按标量处理）；
  - 与input的shape完全相同，且input、values的元素总数均大于0（图模式不支持元素总数为0的空Tensor）。
- 图模式（算子IR）下kernel内部不做broadcast，仅按标量或相同线性偏移读取values，因此不支持input与values的一般广播关系（例如input为0维标量、values为多维张量）。


## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_heaviside](./examples/test_aclnn_heaviside.cpp) | 通过[aclnnHeaviside](./docs/aclnnHeaviside.md)接口方式调用Heaviside算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/heaviside_proto.h)构图方式调用Heaviside算子。 |
