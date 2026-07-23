# SoftshrinkGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：[Softshrink](../softshrink/README.md)的反向算子，计算反向传播的梯度。含NaN保护：当input_x为NaN时，梯度直接传递（与PyTorch行为对齐）。

- 计算公式：

  $$
  SoftshrinkBackward(x, grad) =
  \begin{cases}
  &grad, &if \ |x| > \lambda \\
  &0, &otherwise
  \end{cases}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 330px">
  <col style="width: 280px">
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
      <td>input_grad</td>
      <td>输入</td>
      <td>反向传播过程中上一步输出的梯度，作为本反向算子的输入，公式中的grad。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_x</td>
      <td>输入</td>
      <td>前向Softshrink的输入张量，公式中的x。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambd</td>
      <td>输入</td>
      <td>Softshrink的阈值参数，默认值为0.5，需满足lambd ≥ 0。公式中的λ，aclScalar类型。</td>
      <td>fp32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_y</td>
      <td>输出</td>
      <td>反向传播梯度输出，公式中的SoftshrinkBackward(x, grad)。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- input_grad与input_x的dtype和shape必须一致。
- 支持0-8维ND格式。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_soft_shrink_grad.cpp](./examples/test_aclnn_soft_shrink_grad.cpp) | 通过[aclnnSoftshrinkBackward](./docs/aclnnSoftshrinkBackward.md)接口方式调用SoftshrinkGrad算子。 |
