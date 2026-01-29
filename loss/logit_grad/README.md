# LogitGrad

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
[aclnnLogit](../logit/docs/aclnnLogit.md)的反向传播。
- 计算公式：

$$
dx_i=
\begin{cases} 
NaN, & \text{if } x < \text0 \text{ or } x > 1 ,eps <0 \\
0, & \text{if } x < \text{eps} \text{ or } x > 1 - \text{eps},eps \geq 0 \\
\frac{dy_i}{x_i \cdot (1 - x_i)}, & \text{if } \text{eps} \leq x_i \leq 1 - \text{eps} \\
\end{cases}
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 908px"><colgroup>
  <col style="width: 91px">
  <col style="width: 149px">
  <col style="width: 296px">
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
      <td>公式中的x。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>公式中的dy。</td>
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
      <td>公式中的dx。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- eps的取值对x和输出的影响：
  - eps小于0时，x取值范围不在[0, 1]，输出NaN。
  - eps大于等于0时，x取值范围不在[eps, 1 - eps]，输出0。
  - eps大于1时，输出NaN，取值为1时为Inf。
  - eps取值为Inf时，输出0。
  - eps取值为NaN时，输出NaN。
- x取值对输出的影响：
  - x取值0和1时，输出Inf。
  - x为Inf或NaN时，输出为NaN。
- dy为Inf或NaN时，输出为Inf或NaN。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_logit_grad](./examples/test_aclnn_logit_grad.cpp) | 通过[aclnnLogitGrad](./docs/aclnnLogitGrad.md)接口方式调用LogitGrad算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/logit_grad_proto.h)构图方式调用LogitGrad算子。 |