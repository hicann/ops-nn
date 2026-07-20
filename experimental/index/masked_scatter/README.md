# MaskedScatter

<!-- codespell:ignore TreamTik -->

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品|√|

## 功能说明

- 算子功能：根据布尔掩码mask，将输入tensor中mask为true的位置替换为updates中的值。

- 计算公式：
  给定输入张量$x$，布尔掩码张量$mask$和更新张量$updates$。将$x$和$mask$按一维扁平顺序展开，定义$cnt(i)$为$mask$在位置$i$之前为true的元素个数：

  $$
  cnt(i)=\sum_{k=0}^{i-1}[mask_k=true]
  $$

  则输出张量$y$满足：

  $$
  y_i =
  \begin{cases}
  updates_{cnt(i)}, & mask_i=true \\
  x_i, & mask_i=false
  \end{cases}
  $$

- 示例：
    假设输入张量$x=[1,2,3,4,5,6,7,8]$，掩码张量$mask=[true,false,true,false,true,false,true,false]$，更新张量$updates=[10,20,30,40]$，那么输出张量$y=[10,2,20,4,30,6,40,8]$，具体计算过程如下：
    $$
    \begin{aligned}
    y_0&=updates_0=10 \\
    y_1&=x_1=2 \\
    y_2&=updates_1=20 \\
    y_3&=x_3=4 \\
    y_4&=updates_2=30 \\
    y_5&=x_5=6 \\
    y_6&=updates_3=40 \\
    y_7&=x_7=8
    \end{aligned}
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 60px">
  <col style="width: 60px">
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 60px">
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
      <td>FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、BOOL、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输入</td>
      <td>公式中的mask。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>公式中的updates。</td>
      <td>FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、BOOL、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的y。</td>
      <td>FLOAT、FLOAT16、UINT8、INT8、INT16、INT32、BOOL、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x、mask、y的shape必须相同。
- x、updates、y的数据类型必须相同。
- mask的数据类型必须为BOOL。
- updates按照mask为true的位置顺序依次消耗。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_masked_scatter.cpp](./examples/test_aclnn_masked_scatter.cpp) | 通过[aclnnMaskedScatter](./docs/aclnnMaskedScatter.md)文档中描述的 `aclnnInplaceMaskedScatter` 两段式接口调用 MaskedScatter 算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| TreamTik | 个人开发者 | MaskedScatter | 2026/06/24 | MaskedScatter 算子适配开源仓 |
