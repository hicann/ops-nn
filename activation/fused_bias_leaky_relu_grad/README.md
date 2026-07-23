# FusedBiasLeakyReluGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：BiasAdd + LeakyReLU + Scale三合一反向梯度算子，来自MMCV框架的FusedBiasLeakyReLU反向。给定上游梯度y_grad和前向特征值features，根据features的符号生成梯度掩码，与y_grad逐元素相乘后再乘以scale缩放因子，输出输入特征的梯度x_grad。

- 计算公式：

  $$
  x\_grad[i] = scale \cdot y\_grad[i] \cdot \begin{cases} 1.0 & \text{if } features[i] > 0 \\ negative\_slope & \text{if } features[i] \leq 0 \end{cases}
  $$

## 参数说明

<table style="table-layout: fixed; width: 100%"><colgroup>
<col style="width: 15%">
<col style="width: 12%">
<col style="width: 35%">
<col style="width: 20%">
<col style="width: 18%">
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
    <td>y_grad</td>
    <td>输入</td>
    <td>上游回传的梯度张量，数据类型需与features一致。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>features</td>
    <td>输入</td>
    <td>前向特征值张量，用于符号判断（features > 0作为梯度掩码），shape需与y_grad广播兼容。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>negative_slope</td>
    <td>属性</td>
    <td>LeakyReLU负半轴斜率系数，默认值0.2。</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>scale</td>
    <td>属性</td>
    <td>方差保持缩放因子，默认值sqrt(2) ≈ 1.414213562373。</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>x_grad</td>
    <td>输出</td>
    <td>输入特征的梯度，shape为y_grad和features广播后的结果。</td>
    <td>同y_grad</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- y_grad和features的数据类型必须相同。
- y_grad和features的shape必须满足广播规则，广播后的rank不超过8。
- 支持空Tensor（元素数为0时返回空输出）。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|---|---|---|
| 图模式 | [test_geir_fused_bias_leaky_relu_grad](./examples/test_geir_fused_bias_leaky_relu_grad.cpp) | 通过[算子IR](./op_graph/fused_bias_leaky_relu_grad_proto.h)构图方式调用FusedBiasLeakyReluGrad算子。 |
