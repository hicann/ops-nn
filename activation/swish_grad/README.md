# SwishGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term> |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：Swish激活函数的反向传播，用于计算Swish激活函数的梯度。  
- 计算公式：
  
  - Swish函数公式

  $$
  s(x) = x*\sigma(\beta x)
  $$

  - Swish函数公式的导数实现

  $$

  s^\prime(x)= \beta s(x)+\sigma(\beta x)(1-\beta s(x))= \sigma(\beta x)*(1+\beta x(1-\sigma(\beta x)))

  $$    

  $$
  gradInput = gradOutput * s^\prime(x)
  $$

  $$
  \sigma(x) = {\frac{1} {1+{e}^{-x}}}
  $$

  其中$\sigma(x)$为Sigmoid函数，$s(x)$为Swish函数，$s^\prime(x)$为Swish函数的导数。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>grad</td>
      <td>输入</td>
      <td>表示Swish激活函数正向输出的梯度，公式中的gradOutput。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示用于计算激活函数的张量，公式中的x。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>表示Swish激活函数的正向输出结果，公式中的σ(x)。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>输出</td>
      <td>表示Swish正向输入的梯度值，即对输入进行求导后的结果。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>可选属性</td>
      <td>表示可调节参数，用于控制Swish函数的形状和斜率的标量，公式中的β。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明
无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_swish_grad](examples/test_aclnn_swish_grad.cpp) | 通过[aclnnSwishBackward](docs/aclnnSwishBackward.md)接口方式调用SwishGrad算子。 |