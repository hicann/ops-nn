# GroupNormalizationGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | √ |

## 功能说明

- 算子功能：计算 Group Normalization 的输入梯度。
- 计算公式：

输入按 `[N, G, M]` 布局组织，单 group 的计算公式为：

$$
\hat{x} = (x - mean) \cdot rstd
$$

$$
s_1 = \sum(dy \cdot gamma), \quad s_2 = \sum(dy \cdot gamma \cdot \hat{x})
$$

$$
dx = \frac{rstd}{M} \cdot gamma \cdot (M \cdot dy - s_1 - \hat{x} \cdot s_2)
$$

其中 $x$ 为前向输入，$dy$ 为上游梯度，$gamma$ 为已广播到 `[N, G, M]` 的缩放系数，$mean$ 为每个 group 的均值（形状 `[N, G]`），$rstd$ 为每个 group 的标准差倒数（形状 `[N, G]`），$dx$ 为输出输入梯度（形状与 $x$ 相同）。

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
      <td>前向输入，公式中的 x。</td>
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
      <td>gamma</td>
      <td>输入</td>
      <td>已广播到 [N, G, M] 的缩放系数。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>每个 group 的均值，形状为 [N, G]。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输入</td>
      <td>每个 group 的标准差倒数，形状为 [N, G]。</td>
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

- 输入 x、dy、gamma 和输出 dx 的数据类型需保持一致。
- 输入 x、dy、gamma 的 shape 需相同。
- 输入维度至少为 3 维（[N, G, M] 布局）。
- mean 和 rstd 需为 [N, G] 形状。

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Zhou Jianhua <@LePenseur> | AISS Group, Harbin Institute of Technology (HIT) | GroupNormalizationGrad | 2026/05/12 | GroupNormalizationGrad 算子架构迁移至标准 AscendC 框架 |
