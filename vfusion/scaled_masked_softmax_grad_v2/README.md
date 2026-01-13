# ScaledMaskedSoftmaxBackward

## 支持的产品型号

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：softmax的反向传播，并对结果进行缩放以及掩码。
- 计算公式：

  $$
  out = gradOutput \cdot output - sum(gradOutput \cdot output)\cdot output \\
  out = \begin{cases}
  out * scale, &\text { mask is 0 } \\
  0, &\text { mask is 1 }
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
      <td>输入</td>
      <td>公式中的输入gradOutput。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>公式中的输入y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输入</td>
      <td>公式中的输入mask。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>属性</td>
      <td>默认值为1.0。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fixTriuMask</td>
      <td>属性</td>
      <td>仅支持false</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的out。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_scaled_masked_softmax_backward](./examples/test_aclnn_scaled_masked_softmax_backward.cpp) | 通过[aclnnScaledMaskedSoftmaxBackward](./docs/aclnnScaledMaskedSoftmaxBackward.md)接口方式调用aclnnScaledMaskedSoftmaxBackward算子。    |
