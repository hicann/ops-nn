# SigmoidGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------- |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> | √        |

## 功能说明

- 算子功能：完成[sigmoid](../../activation/sigmoid_grad/docs/aclnnSigmoidBackward.md)的反向传播，根据sigmoid反向传播梯度与正向输出计算sigmoid的梯度输入。
- 计算公式：

  $$
  out = {\frac{1} {1+{e}^{-input}}}
  $$

  $$
  grad\_input = grad\_output * \sigma(x) * (1 - \sigma(x))
  $$

其中$\sigma(x)$为sigmoid函数的正向输出，$\sigma(x)*(1-\sigma(x))$为sigmoid函数的导数。

 **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1330px"><colgroup>
  <col style="width: 101px">
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
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NC1HWC0、ND、FRACTAL_NZ、C1HWNCoC0</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输入</td>
      <td>前向sigmoid的输出，公式中的out。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要与grad_output，grad_input一致。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NC1HWC0、ND、FRACTAL_NZ、C1HWNCoC0</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td>为self的梯度值，公式中的grad\_input。</td>
      <td>数据类型、shape需要与grad_output，output一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NC1HWC0、ND、FRACTAL_NZ、C1HWNCoC0</td>
      <td>1-8</td>
      <td>√</td>
     </tr>
   </tbody>
  </table>
  
## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sigmoid_grad](./examples/test_aclnn_sigmoid_grad.cpp) | 通过[aclnnSigmoidBackward](./docs/aclnnSigmoidBackward.md)接口方式调用SigmoidGrad算子。    |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| hth810 | 个人开发者 | SigmoidGrad | 2025/11/28 | SigmoidGrad算子适配开源仓 |
