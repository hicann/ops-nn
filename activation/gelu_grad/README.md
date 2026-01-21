# GeluGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 接口功能：[Gelu](../gelu/README.md)激活函数的反向传播算子，用于计算Gelu激活函数的梯度。
- 计算公式：
  - 前向计算公式
  $$
  out=GELU(self)=self × Φ(self)=0.5 * self * (1 + tanh( \sqrt{2 / \pi} * (self + 0.044715 * self^{3})))
  $$
  - 反向梯度公式
  $$
   out = \frac{d(\text{GELU})}{dx} = dy \cdot \left[ \underbrace{0.5(1 + \tanh(\text{inner}))}_{\text{left\_derivative}}+ \underbrace{0.5x\cdot (1-\tanh^2(\text{inner})) \cdot \beta(1+3\cdot 0.044715x^2)}_{\text{right\_derivative}}\right]

  $$
  - 其中：
  $$
  
   \beta = \sqrt{\frac{2}{\pi}},\quad \text{inner}= \beta \left(x+0.044715x^3 \right)
  $$
  
## 参数说明

<table style="undefined;table-layout: fixed; width: 919px"><colgroup>
  <col style="width: 130px">
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
      <td>dy</td>
      <td>输入</td>
      <td>这是损失函数对GELU输出的梯度，公式中的dy。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>这是函数的输入，公式中的x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>GELU函数的输出，即GELU(x)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>输出</td>
      <td>表示gelu_grad函数的输出，对应公式中的out。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_gelu_backward](./examples/test_aclnn_gelu_backward.cpp) | 通过[aclnnGelu](./docs/aclnnGeluBackward)接口方式调用GeluGrad算子。    |
