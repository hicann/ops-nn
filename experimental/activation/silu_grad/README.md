# SiluGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | √ |

## 功能说明

- 算子功能：求SiLU（也称Swish，β=1）函数的梯度，用于SiLU激活函数的反向传播。

- 计算公式：

  - SiLU函数公式

$$
y = x \cdot \sigma(x)
$$

  - SiLU函数的导数

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$
y' = \sigma(x) \cdot (1 + x \cdot (1 - \sigma(x)))
$$

$$
dx = dy \cdot \sigma(x) \cdot (1 + x \cdot (1 - \sigma(x)))
$$

其中$\sigma(x)$为Sigmoid函数，$y$为SiLU函数，$y'$为SiLU函数的导数。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
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
      <td>dy</td>
      <td>输入</td>
      <td>反向传播过程中上一步输出的梯度，公式中的dy。</td>
      <td>FLOAT16、FLOAT、BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>正向的输入数据，公式中的x。</td>
      <td>FLOAT16、FLOAT、BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>SiLU反向计算的梯度输出，公式中的dx。</td>
      <td>FLOAT16、FLOAT、BF16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入dy与x的shape需一致，且与输出dx的shape一致（不支持broadcast的场景下）；支持broadcast的场景下，dy与x需满足broadcast关系，dx的shape需与broadcast后的shape一致。
- 输入dy、x与输出dx的数据类型需一致。
- 输入维度不超过8维。
- BF16数据类型仅在部分产品型号上支持。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_silu_grad.cpp](./examples/test_aclnn_silu_grad.cpp) | 通过[aclnnSiluBackward](./docs/aclnnSiluBackward.md)接口方式调用SiluGrad算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
|王正阳  |  北京交通大学-赵宏智老师团队| SiluGrad |2026/06/04  | SiluGrad算子适配开源仓 |
