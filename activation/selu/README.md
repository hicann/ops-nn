# Selu

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：对输入Tensor逐元素计算SELU（Scaled Exponential Linear Unit）激活函数。
- 计算公式：

  $$
  \text{SELU}(x) = \text{scale} \times \left( \max(0, x) + \min\left(0, \alpha \times \left(e^x - 1\right)\right) \right)
  $$

  其中：

  - $\alpha = 1.6732632423543772848170429916717$
  - $\text{scale} = 1.0507009873554804934193349852946$

  分段行为：

  - 当$x > 0$时：$\text{SELU}(x) = \text{scale} \times x$（线性区）
  - 当$x \le 0$时：$\text{SELU}(x) = \text{scale} \times \alpha \times (e^x - 1)$（指数饱和区）

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>公式中的输入x。</td>
    <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT8</td>
    <td>ND</td>
    <td>1-8</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>公式中的输出y，shape与输入x相同。</td>
    <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT8</td>
    <td>ND</td>
    <td>1-8</td>
  </tr>
</tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_selu](./examples/test_aclnn_selu.cpp) | 通过[aclnnSelu&aclnnInplaceSelu](./docs/aclnnSelu&aclnnInplaceSelu.md)接口方式调用Selu算子。 |
| aclnn调用 | [test_aclnn_inplace_selu](./examples/test_aclnn_inplace_selu.cpp) | 通过[aclnnSelu&aclnnInplaceSelu](./docs/aclnnSelu&aclnnInplaceSelu.md)接口方式调用Selu算子。 |
