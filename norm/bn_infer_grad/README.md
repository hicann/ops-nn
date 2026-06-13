# BNInferGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：计算BatchNorm在推理模式（training=False）下输入数据关于损失函数的梯度，等价于PyTorch `batch_norm_backward(..., training=false, ...)` 中 `grad_input` 的子计算。

- 计算公式：

$$
x\_backprop = grads \times scale \times \frac{1}{\sqrt{batch\_variance + \epsilon}}
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>grads</td>
      <td>输入</td>
      <td>上游梯度张量，公式中的grads。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND、NCHW、NHWC、NC1HWC0</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>BatchNorm缩放参数gamma，公式中的scale，长度等于grads通道维C。</td>
      <td>fp32</td>
      <td>ND、NCHW、NHWC、NC1HWC0</td>
    </tr>
    <tr>
      <td>batch_variance</td>
      <td>输入</td>
      <td>推理阶段BatchNorm方差，公式中的batch_variance，长度等于grads通道维C。</td>
      <td>fp32</td>
      <td>ND、NCHW、NHWC、NC1HWC0</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>属性</td>
      <td>防除零常数，公式中的ε，默认值为0.0001。</td>
      <td>fp32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x_backprop</td>
      <td>输出</td>
      <td>BNInferGrad计算的输出张量，shape与grads一致。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND、NCHW、NHWC、NC1HWC0</td>
    </tr>
  </tbody></table>

## 约束说明

- grads与x_backprop的dtype须保持一致。
- scale与batch_variance的dtype须为fp32。
- 当format为NC1HWC0时，scale与batch_variance的shape需调整为(1, C1, 1, 1, C0)。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式  | [test_geir_bn_infer_grad.cpp](examples/test_geir_bn_infer_grad.cpp) | 通过图模式方式调用BNInferGrad算子。 |
