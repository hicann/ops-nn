# Celu

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- **算子功能**：Celu（Continuously Differentiable Exponential Linear Unit）是一种连续可微的激活函数，对正输入返回固定值，对负输入返回指数衰减曲线。
- **计算公式**：

  $$
  y = \begin{cases}
  \alpha_3 \times 3 & \text{if } x \geq 0 \\
  \alpha_1 \times \left(\exp\left(\frac{x}{\alpha_2}\right) - 1\right) & \text{if } x < 0
  \end{cases}
  $$

  其中：
  - 当`x >= 0`时，输出为常数`alpha3 * 3`
  - 当`x < 0`时，输出为指数衰减形式`alpha1 * (exp(x / alpha2) - 1)`
- **精度保护**：针对`exp`操作的溢出风险，对负区输入执行Mins截断保护（FP16截断阈值为11.0，FP32截断阈值为87.0），防止指数运算溢出。

## 参数说明

<table style="table-layout: fixed; width: 100%"><colgroup>
<col style="width: 12%">
<col style="width: 12%">
<col style="width: 32%">
<col style="width: 24%">
<col style="width: 20%">
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
    <td>公式中的输入x，任意形状的张量。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>alpha1</td>
    <td>属性</td>
    <td><ul><li>负区指数曲线的缩放系数，对应公式中的alpha1。</li><li>默认值为1.0。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>alpha2</td>
    <td>属性</td>
    <td><ul><li>负区指数曲线的斜率系数，对应公式中的alpha2，必须大于0。</li><li>默认值为1.0。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>alpha3</td>
    <td>属性</td>
    <td><ul><li>正区的固定输出值，对应公式中的alpha3。</li><li>默认值为1.0。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>公式中的输出y，与输入x形状相同的张量。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- 本算子仅支持<term>Ascend 950PR/Ascend 950DT</term>（DAV_3510/arch35），不支持其他芯片。
- 输入tensor支持的数据类型为FLOAT16和FLOAT，输出与输入数据类型一致。
- 输入tensor的格式必须为ND。
- alpha2属性必须大于0，否则可能导致除零或数值不稳定。
- 输入支持任意shape（包括0元素空tensor），算子内部已处理空tensor边界情况。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_celu](./examples/test_geir_celu.cpp) | 通过[算子IR](./op_graph/celu_proto.h)构图方式调用Celu算子。 |
