# TanhGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | √ |

## 功能说明

- 算子功能：计算 Tanh 的梯度。
- 计算公式：

$$
dx = dy * (1 - y * y)
$$

其中 $y$ 为 Tanh 前向输出，$dy$ 为上游梯度，$dx$ 为输入梯度。

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
      <td>y</td>
      <td>输入</td>
      <td>Tanh 前向输出，公式中的 y。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>上游梯度，公式中的 dy。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>输入梯度，公式中的 dx。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入 y、dy 和输出 dx 的数据类型需保持一致。
- 输入 y 和 dy 的 shape 需相同。

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Zhou Jianhua <@LePenseur> | AISS Group, Harbin Institute of Technology (HIT) | TanhGrad | 2026/05/12 | TanhGrad 算子架构迁移至标准 AscendC 框架 |
