# Prelu

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

- 算子功能：计算输入张量的 PReLU 值。当输入元素大于 0 时输出该元素本身；当输入元素小于等于 0 时输出该元素与 `weight` 的乘积。

- 计算公式：

$$
{y}_i =
\begin{cases}
x_i, & x_i > 0 \\
x_i \times weight, & x_i \le 0
\end{cases}
$$

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
      <td>x</td>
      <td>输入</td>
      <td>待进行 Prelu 计算的输入张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>Prelu 负半轴权重，元素个数为 1 或与输入通道数一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>Prelu 计算后的输出张量，shape 与 x 一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `x`、`weight`、`y` 的数据类型需要一致。
- `y` 的 shape 必须与 `x` 完全一致。
- `weight` 的元素个数必须为 1 或与 `x` 的通道数一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_prelu.cpp](./examples/test_aclnn_prelu.cpp) | 通过 [aclnnPrelu](./docs/aclnnPrelu.md) 接口方式调用 Prelu 算子。 |

## 贡献说明

|  贡献者  | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ----- | ---- | ---- | ---- | ---- |
|Han Chuang| 个人开发者 | Prelu | 2026/06/05 | Prelu 算子适配开源仓 |
