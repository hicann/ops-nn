# BinaryCrossEntropyGrad

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

- 算子功能：求二元交叉熵反向传播的梯度值。
- 计算公式:

  二元交叉熵的计算公式如下：
  
  $$
  z_i = - weight_i \cdot (y_i \cdot ln(x_i) + (1 - y_i) \cdot ln(1- x_i))
  $$

  其中$x$表示网络前一层的输出，即正向的预测值；$y$表示样本的标签值，求二元交叉熵对$x$的偏导：

  $$
  \begin{aligned}
  \frac {\partial z}{\partial x} &= -weight \cdot \frac {\partial ((y_i \cdot ln(x_i) + (1-y_i) \cdot ln(1-x_i))}{\partial x} \\
  &= -weight \cdot (\frac {\partial (y \cdot ln(x))}{\partial x} + \frac {\partial ((1-y) \cdot ln(1-x))}{\partial x}) \\
  &= -weight \cdot (\frac {y}{x} - \frac {(1-y)}{1-x}) \\
  &= -weight \cdot \frac {y(1-x)-x(1-y)}{x(1-x)} \\
  &= -weight \cdot \frac {y-x}{x(1-x)} \\
  &= weight \cdot \frac {x-y}{x(1-x)}
  \end{aligned}
  $$

  $$
  out = grad\_output \cdot weight \cdot \frac {x-y}{x(1-x)} \\
  out = mean(grad\_input) \ if \ reduction = mean
  $$


## 参数说明

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>网络正向前一层的计算结果。数据类型需要与其它参数一起转换到promotion类型。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>样本的标签值。数据类型需要与其它参数一起转换到promotion类型，shape可以broadcast到x的shape。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_output</td>
      <td>输入</td>
      <td>网络反向传播前一步的梯度值。数据类型需要与其它参数一起转换到promotion类型，shape可以broadcast到x的shape。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>结果的权重。数据类型需要与其它参数一起转换到promotion类型，shape可以broadcast到x的shape。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>存储梯度计算结果，shape与x相同。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>属性</td>
      <td>表示对二元交叉熵反向求梯度计算结果做的reduce操作。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>

  </tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_binary_cross_entropy_backward.cpp](examples/test_aclnn_binary_cross_entropy_backward.cpp) | 通过[aclnnBinaryCrossEntropyBackward](docs/aclnnBinaryCrossEntropyBackward.md)接口方式调用BinaryCrossEntropyGrad算子。 |