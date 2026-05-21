# Gelu

## 产品支持情况

|| 产品                                                         | 是否支持 |
|| :----------------------------------------------------------- | :------: |
|| <term>Atlas A2 训练系列产品/Atlas 800I A2推理产品</term>     |    √     |

## 功能说明

- 算子功能：逐元素计算张量的Gelu(Gaussian Error Linear Unit)激活函数。
- 计算公式为：

$$
\text{Gelu}(x) = x \cdot \Phi(x) = \frac{x}{2} \left[ 1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right) \right]
$$

近似计算公式（Tanh近似）：

$$
\text{Gelu}(x) \approx \frac{x}{1 + \exp\left(-1.595769122 \cdot (x + 0.0455399241 \cdot x^3)\right)}
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
  <col style="width: 120px">
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
      <td>公式中的输入张量x</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量y</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 调用说明

|| 调用方式 | 调用样例                                                                   | 说明                                                             |
||--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
|| aclnn调用 | [test_aclnn_gelu](./examples/test_aclnn_gelu.cpp) | 通过aclnnGelu接口方式调用Gelu算子。    |

## 贡献说明

|| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|| ---- | ---- | ---- | ---- | ---- |
|| fulltower | 个人开发者 | Gelu | 2026/05/18 | 新增Gelu算子 |
