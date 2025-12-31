# Swish

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> |√|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：计算Swish激活函数值。

- 计算公式：
  $$
  y = \frac{x}{1 + e^{-\text{scale} \times x}}
  $$
  其中：
  - $x$ 表示输入张量
  - $\text{scale}$ 表示缩放参数，默认值为1.0
  - $e$ 表示自然对数的底数
  - $y$ 表示输出张量

## 参数说明

<table style="undefined;table-layout: fixed; width: 1480px">
  <colgroup>
    <col style="width: 177px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 292px">
    <col style="width: 152px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示输入张量，支持1D-8D维度</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示输出张量，与输入x具有相同的类型、形状和格式</td>
      <td>与x一致</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>属性</td>
      <td>表示缩放参数，用于调整Swish函数的形状。默认值为1.0</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_swish](./examples/test_aclnn_swish.cpp) | 通过[aclnnSwish](./docs/aclnnSwish.md)接口方式调用Swish算子。    |
