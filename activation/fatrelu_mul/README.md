# FatreluMul

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：

  将输入Tensor按照最后一个维度分为左右两个Tensor：x1和x2，对左边的x1进行Threshold计算，将计算结果与x2相乘。

- 计算公式：

  给定输入张量input，最后一维的长度为2d，进行以下计算：

  1. 将input分割为两部分：

     $$
     x_1 = \text{input}[..., :d], \quad x_2 = \text{input}[..., d:]
     $$

  2. 对x1应用Threshold激活函数，定义如下：

     $$
     \text{Threshold}(x, \text{threshold}) =
        \begin{cases}
        0 & \text{if } x < \text{threshold} \\
        x & \text{if } x \geq \text{threshold}
        \end{cases}
     $$

     因此，计算：

     $$
     x_1 = \text{Threshold}(x_1, \text{threshold})
     $$

  3. 最终输出是x1和x2的逐元素乘积：

     $$
     \text{out} = x_1 \times x_2
     $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 891px"><colgroup>
  <col style="width: 121px">
  <col style="width: 144px">
  <col style="width: 253px">
  <col style="width: 257px">
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
      <td>input</td>
      <td>输入</td>
      <td>公式中的输入input。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>输入</td>
      <td><ul><li>公式中的输入threshold。</li><li>threshold不支持NaN和Inf。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
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

典型场景尾轴为16的倍数，当尾轴为非32Byte对齐时，建议走小算子拼接逻辑。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_fatrelu_mul](./examples/test_aclnn_fatrelu_mul.cpp) | 通过[aclnnFatreluMul](./docs/aclnnFatreluMul.md)接口方式调用FatreluMul算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/fatrelu_mul_proto.h)构图方式调用FatreluMul算子。 |
