# GeluMul

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：

  将输入Tensor按照最后一个维度分为左右两个Tensor：x1和x2，对左边的x1进行GELU计算，将计算结果与x2相乘。

- 计算公式：
  
  给定输入张量 `input`，最后一维的长度为 `2d`，函数 `GeluMul` 进行以下计算：

  1. 将 `input` 分割为两部分：

     $$
     x_1 = \text{input}[..., :d], \quad x_2 = \text{input}[..., d:]
     $$

  2. 对x1应用GELU激活函数，"tanh"模式公式如下：

     $$
     \text{GELU}(x) = 0.5 \cdot x \cdot \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \cdot \left( x + 0.044715 x^3 \right) \right) \right)
     $$

     “none”对应的erf模式公式如下：

     $$
     \text{GELU}(x) = 0.5 \cdot x \left( 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right)
     $$

     因此，计算：

     $$
     x_1 = \text{GELU}(x_1)
     $$

  3. 最终输出是x1和x2的逐元素乘积：

     $$
     \text{out} = x_1 \times x_2
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
      <td>input</td>
      <td>输入</td>
      <td>公式中的输入input。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>approximateOptional</td>
      <td>可选属性</td>
      <td>GELU计算的模式，只支持“none”和“tanh”，分别对应GELU的erf模式和tanh模式，输入为空指针时为“none”。</td>
      <td>STRING</td>
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

典型场景尾轴为16的倍数，当尾轴为非32B对齐时，建议走小算子拼接逻辑。 

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_gelu_mul](./examples/test_aclnn_gelu_mul.cpp) | 通过[aclnnGeluMul](./docs/aclnnGeluMul.md)接口方式调用GeluMul算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/gelu_mul_proto.h)构图方式调用GeluMul算子。 |