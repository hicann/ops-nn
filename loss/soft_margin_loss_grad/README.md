# SoftMarginLossGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：求SoftMarginLoss反向传播的梯度值。
- 计算公式：

  SoftMarginLoss的前向计算公式如下：

  $$
  loss_i = ln(1 + exp(-self_i \cdot target_i))
  $$

  对 $self$ 求偏导得到反向梯度：

  $$
  out = cof \cdot \frac{-target \cdot exp(-self \cdot target)}{1 + exp(-self \cdot target)} \cdot grad\_output
  $$

  其中reduction为mean时 $cof = 1/N$，否则 $cof = 1.0$。

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
      <td>self</td>
      <td>输入</td>
      <td>网络正向前一层的预测值。数据类型需要与其它参数一起转换到promotion类型。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td>样本的标签值。数据类型需要与其它参数一起转换到promotion类型，shape可以broadcast到self的shape。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_output</td>
      <td>输入</td>
      <td>网络反向传播前一步的梯度值。数据类型需要与其它参数一起转换到promotion类型，shape可以broadcast到self的shape。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>存储梯度计算结果，shape为self、target、grad_output广播后的shape。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>属性</td>
      <td>表示对反向求梯度计算结果做的reduce操作，取值为"none"/"mean"/"sum"，默认"mean"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_soft_margin_loss_grad.cpp](examples/arch35/test_aclnn_soft_margin_loss_grad.cpp) | 通过[aclnnSoftMarginLossBackward](docs/aclnnSoftMarginLossBackward.md)接口方式调用SoftMarginLossGrad算子。 |
