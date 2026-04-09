# HardtanhGrad

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 功能说明

- 算子功能：计算Hardtanh激活函数的梯度（反向传播）。

- 计算公式：

  $$
  res_{i} = grad\_output_{i} \times grad\_self_{i}
  $$

  $$
  grad\_self_{i} = 
  \begin{cases}
    0,\ \ \ \ \ \ \ if \ \ self_{i}>max \\
    0,\ \ \ \ \ \ \  if\ \ self_{i}<min \\
    1,\ \ \ \ \ \ \ \ \ \ \ \ otherwise \\
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
      <td>gradOutput</td>
      <td>输入</td>
      <td>待进行hardtanh_grad计算的入参，公式中的gradOutput。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>待进行hardtanh_grad计算的入参，公式中的self。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>min</td>
      <td>输入属性</td>
      <td>待进行hardtanh_grad计算的入参，公式中的min。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>  
    <tr>
      <td>max</td>
      <td>输入属性</td>
      <td>待进行hardtanh_grad计算的入参，公式中的max。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行hardtanh_grad计算的出参，公式中的res。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输出的数据类型和输入一样。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_hardtanh_backward.cpp](./examples/test_aclnn_hardtanh_backward.cpp) | 通过[aclnnHardtanhBackward](./docs/aclnnHardtanhBackward.md)接口方式调用HardtanhGrad算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| infinity | 个人开发者 | HardtanhGrad | 2026/01/16 | HardtanhGrad算子适配开源仓 |
