# Logit

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>昇腾910_95 AI处理器</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：

  该算子是概率到对数几率（log-odds）转换的一个数学运算，常用于概率值的反变换。它可以将一个介于0和1之间的概率值转换为一个实数值。

- 计算公式：

  $$
  y_i = \ln \left( \frac{z_i}{ 1 -z_i} \right)
  $$

  其中：

  $$
  z_i = 
  \begin{cases} 
  \text{eps}, & \text{if } x_i < \text{eps} \\
  x_i, & \text{if } \text{eps} \leq x_i \leq 1 - \text{eps} \\
  1 - \text{eps}, & \text{if } x_i > 1 - \text{eps}
  \end{cases}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 878px"><colgroup>
  <col style="width: 91px">
  <col style="width: 149px">
  <col style="width: 266px">
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
      <td>eps</td>
      <td>属性</td>
      <td><ul><li>输入x的eps限制边界，防止除0错误。</li><li>建议值-1。</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- input取值范围不在[0, 1]，输出NaN，取值0时为-Inf，取值为1时为Inf。
- eps大于1时，输出NaN，取值为1时为Inf。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_logit](./examples/test_aclnn_logit.cpp) | 通过[aclnnLogit](./docs/aclnnLogit.md)接口方式调用Logit算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/logit_proto.h)构图方式调用Logit算子。 |
