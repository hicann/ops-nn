# HardSwishGradV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：[aclnnHardswish](../hard_swish/docs/aclnnHardswish&aclnnInplaceHardswish.md)的反向传播，完成张量self的梯度计算。
- 算子功能差异点说明：相比于[aclnnHardswishBackward](../hard_swish_grad/docs/aclnnHardswishBackward.md)接口，aclnnHardswishBackwardV2的计算公式做了细微调整，详见“计算公式”。
- 计算公式：

  $$
  out_{i} = gradOutput_{i} \times gradSelf_{i}
  $$

  其中 gradSelf 的计算公式为

  $$
  gradSelf_{i} = \begin{cases}
  0, & self_{i} \le -3, \\
  self_{i} / 3 + 0.5, & -3 \lt self_{i} \lt 3, \\
  1, & self_{i} \ge 3
  \end{cases}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>表示输入张量，公式中的输入gradOutput。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td><ul><li>输入数据。公式中的self。</li><li>支持空Tensor。</li><li>shape需要与gradOutput、out相同。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示输出张量，公式中的out。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    </tbody>
  </table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_hard_swish_grad_v2](./examples/test_aclnn_hard_swish_backward_v2.cpp) | 通过[aclnnHardswishBackwardV2](./docs/aclnnHardswishBackwardV2.md)接口方式调用HardswishBackwardV2算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/hard_swish_grad_v2_proto.h)构图方式调用HardswishBackwardV2算子。 |
