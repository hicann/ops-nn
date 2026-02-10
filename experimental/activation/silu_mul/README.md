# SiluMul

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：

  对输入Tensor x进行Silu激活计算，将计算结果与输入Tensor y相乘。

- 计算公式：

  给定输入张量 `x` 和 `y`，函数 `SiluMul` 进行以下计算：

  1. 对 `x` 应用 SiLU (Sigmoid Linear Unit) 激活函数：
     $$
     \text{SiLU}(x) = x \cdot \text{Sigmoid}(x) = \frac{x}{1 + e^{-x}}
     $$

  2. 最终输出是 SiLU(x) 和 y 的逐元素乘积：
     $$
     \text{out} = \text{SiLU}(x) \times y
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
      <td>x</td>
      <td>输入</td>
      <td>公式中的输入x。</td>
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
| aclnn调用 | [test_aclnn_silu_mul](./examples/test_aclnn_silu_mul.cpp) | 通过[aclnnSiluMul](./docs/aclnnSiluMul.md)接口方式调用SiluMul算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/silu_mul_proto.h)构图方式调用SiluMul算子。 |