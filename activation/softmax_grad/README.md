# SoftmaxGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 接口功能：完成[softmax](../softmax_v2/README.md)的反向传播。

- 计算公式：对于Softmax函数的求导，可以使用以下公式：
  out（输入梯度值）和gradOutput（上一层输出梯度）、output（Softmax正向输出）的关系可表示如下：
  $$
  out = gradOutput \cdot output - sum(gradOutput \cdot output)\cdot output
  $$

- **参数说明：**
   <table style="undefined;table-layout: fixed; width: 1349px"><colgroup>
    <col style="width: 158px">
    <col style="width: 120px">
    <col style="width: 253px">
    <col style="width: 283px">
    <col style="width: 218px">
    <col style="width: 110px">
    <col style="width: 102px">
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
      <td>反向传播的梯度值，即上一层的输出梯度。公式中的gradOutput。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输入</td>
      <td>Softmax函数的输出值，公式中的output。</td>
      <td><ul><li>支持空Tensor。</li><li>shape和数据类型需要与gradOutput保持一致。</li></ul></td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>输入</td>
      <td>Softmax函数的维度。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <tr>
      <td>out</td>
      <td>输出</td>
      <td>函数的输出是输入的梯度值。</td>
      <td><ul><li>对输入进行求导后的结果，支持多种数据类型，不需要额外申请空间，其他数据类型通过自动cast能力支持，但会额外申请空间。</li><li>shape需要与gradOutput保持一致。</li></ul></td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    </tbody>
     </table>
- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT32。 
## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_softmax_grad](./examples/test_aclnn_softmax_grad.cpp) | 通过[aclnnSoftmaxGrad](./docs/aclnnSoftmaxBackward.md)接口方式调用SoftmaxGrad算子。 |
