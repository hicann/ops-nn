# Celu

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

- **算子功能**：Celu（Continuously Differentiable Exponential Linear Unit）是一种连续可微的激活函数，对正输入按系数线性缩放，对负输入返回指数衰减曲线，兼容ONNX的Celu算子。
- **计算公式**：

  $$
  y = \begin{cases}
  \alpha_3 \times x & \text{if } x \geq 0 \\
  \alpha_1 \times \left(\exp\left(\frac{x}{\alpha_2}\right) - 1\right) & \text{if } x < 0
  \end{cases}
  $$

  其中：
  - 当`x >= 0`时，输出为线性缩放`alpha3 * x`（默认`alpha3=1`时即恒等`y=x`，与ONNX Celu一致）
  - 当`x < 0`时，输出为指数衰减形式`alpha1 * (exp(x / alpha2) - 1)`
- **精度保护**：计算使用FP32中间精度（FP16输入先提升为FP32），并将`exp`输入上限统一截断为87.0，以避免FP32指数溢出。

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
    <td>公式中的输入x，ND张量，支持0元素空tensor。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>alpha1</td>
    <td>属性</td>
    <td><ul><li>负区指数曲线的缩放系数，对应公式中的alpha1。</li><li>取值范围为任意FLOAT值。</li><li>默认值为1.0。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>alpha2</td>
    <td>属性</td>
    <td><ul><li>负区指数曲线的斜率系数，对应公式中的alpha2，取值限制见约束说明。</li><li>默认值为1.0。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>alpha3</td>
    <td>属性</td>
    <td><ul><li>正区线性缩放系数，对应公式中的alpha3。</li><li>取值范围为任意FLOAT值。</li><li>默认值为1.0。</li></ul></td>
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

- 输入tensor支持的数据类型为FLOAT16和FLOAT，输出与输入数据类型一致。
- alpha2属性必须大于0；alpha2为0、负值或NaN时校验失败。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_celu](./examples/test_geir_celu.cpp) | 通过[算子IR](./op_graph/celu_proto.h)构图方式调用Celu算子。 |
