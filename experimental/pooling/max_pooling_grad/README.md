<!--
 This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 and is contributed to the CANN Open Software.

 Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 All Rights Reserved.

 Authors (accounts):
 - Zhou Jianhua <@LePenseur>
 - Su Tonghua <@sutonghua>

 This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 CANN Open Software License Agreement Version 2.0 (the "License").
 Please refer to the License for details. You may not use this file except in compliance with the License.
 THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 See LICENSE in the root of the software repository for the full text of the License.
-->

# MaxPoolingGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：最大池化的反向传播，计算输入梯度。

- 计算公式：

  对于非重叠窗口（stride = kernel_size，元素一一对应）:

  $$
  \frac{\partial L}{\partial x_i} =
  \begin{cases}
  \frac{\partial L}{\partial y_i}, & \text{if } x_i = y_i \\
  0, & \text{otherwise}
  \end{cases}
  $$

  其中 $x_i$ 为前向输入元素，$y_i$ 为前向输出（最大值），$\frac{\partial L}{\partial y_i}$ 为上游梯度。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>上游梯度 (upstream gradient)，公式中的 $\frac{\partial L}{\partial y}$。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>前向输入 (original input)，公式中的 $x$。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>前向输出 (pooling result / max values)，公式中的 $y$。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>输入梯度 (gradient w.r.t. input)，公式中的 $\frac{\partial L}{\partial x}$。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 适用于非重叠窗口 (stride = kernel_size) 场景
- x / y / dy / dx 四者形状相同
- `y` 为前向 max pooling 的输出（每个窗口的最大值），已按非重叠窗口展开到与 `x` 相同的形状；本算子在 `x` 与 `y` 同形前提下逐元素计算 `dx = (x == y) ? dy : 0`
- 仅支持 ND 格式
- 不支持 BF16 数据类型

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_max_pooling_grad](./examples/test_aclnn_max_pooling_grad.cpp) | 通过[aclnnMaxPoolingGrad](./docs/aclnnMaxPoolingGrad.md)接口方式调用max_pooling_grad算子。    |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| - | - | MaxPoolingGrad | 2026/05/17 | MaxPoolingGrad算子适配开源仓 |
