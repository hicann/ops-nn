# SigmoidGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |
|  <term>Atlas 200/300/500 推理产品</term>       |     ×    |

## 功能说明

- 算子功能：完成[sigmoid](../../sigmoid/docs/aclnnSigmoid&aclnnInplaceSigmoid.md)的反向传播，根据sigmoid反向传播梯度与正向输出计算sigmoid的梯度输入。
- 计算公式：

  $$
  out = {\frac{1} {1+{e}^{-input}}}
  $$

  $$
  grad\_input = grad\_output * \sigma(x) * (1 - \sigma(x))
  $$

其中$\sigma(x)$为sigmoid函数的正向输出，$\sigma(x)*(1-\sigma(x))$为sigmoid函数的导数。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 230px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOutput</td>
      <td>输入</td>
      <td>损失函数对sigmoid输出的梯度，公式中的grad\_output。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要与output，grad_input一致。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输入</td>
      <td>前向sigmoid的输出，公式中的out。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要与grad_output，grad_input一致。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td>为self的梯度值，公式中的grad\_input。</td>
      <td>数据类型、shape需要与grad_output，output一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
   </tbody>
  </table>

- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                                                         |
| ---------------- | --------------------------- |----------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_sigmoid_grad.cpp](examples/test_aclnn_sigmoid_grad.cpp) | 通过[aclnnSigmoidBackward](docs/aclnnSigmoidBackward.md)接口方式调用SigmoidGrad算子。 |