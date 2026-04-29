# Mish

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas 800I A2推理产品</term>     |    √     |

## 功能说明

- 算子功能：逐元素计算张量的Mish激活函数。
- SUPER_PERFORMANCE模式的计算公式为：
$$y = x*(1-2/(1+(1+(1+x/64)^{64})^2))$$
其他模式的计算公式为：

$$y =
\begin{cases}
x*(2e^{-x} + 1) / (2e^{-2x} + 2e^{-x} + 1),  & \text{if $x > 0$} \\
x*(2e^x + e^{2x}) / (2 + 2e^x + e^{2x}), & \text{if $x \leq 0$}
\end{cases}$$

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

## 约束说明
## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_mish](./examples/test_aclnn_mish.cpp) | 通过aclnnMish接口方式调用Mish算子。    |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| 闻毅 | 个人开发者 | Mish | 2025/12/23 | Mish算子适配开源仓 |
