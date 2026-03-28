# LogSigmoidGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |    √     |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：[LogSigmoid](../../log_sigmoid/docs/aclnnLogSigmoid.md)的反向传播，根据上一层传播的梯度与LogSigmoid正向输入计算其梯度输入。

- 计算公式：

  $$
  \sigma(x) = LogSigmoid(x) = \log(\frac{1}{1+e^{-x}})
  $$

  $$
  gradInput = gradOutput * \sigma^\prime(x) = gradOutput * \frac{1}{1+e^{x}}
  $$

  其中$x$为LogSigmoid函数的正向输入，$\sigma(x)$为LogSigmoid函数，$\sigma^\prime(x)$为LogSigmoid函数的导数。

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1497px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 247px">
  <col style="width: 300px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
  <col style="width: 145px">
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
      <td>表示LogSigmoid激活函数正向输出的梯度，公式中的gradOutput。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>features</td>
      <td>输入</td>
      <td>待进行反向计算的入参，公式中的x。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops</td>
      <td>输出</td>
      <td>出参，公式中的gradInput。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- 输入输出数据类型保持一致。

## 调用示例

| 调用方式   | 样例代码                                                                         | 说明                                                                               |
| ---------------- |------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_logsigmoid_grad.cpp](examples/test_aclnn_logsigmoid_grad.cpp) | 通过[aclnnLogSigmoidBackward](docs/aclnnLogSigmoidBackward.md)接口方式调用LogSigmoidGrad算子。 |
