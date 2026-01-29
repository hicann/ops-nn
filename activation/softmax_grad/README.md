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
  <table style="undefined;table-layout: fixed; width: 813px"><colgroup>
  <col style="width: 103px">
  <col style="width: 144px">
  <col style="width: 194px">
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
      <td>gradOutput</td>
      <td>输入</td>
      <td>反向传播的梯度值，即上一层的输出梯度。公式中的gradOutput。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输入</td>
      <td>Softmax函数的输出值，公式中的output。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>输入</td>
      <td>Softmax函数的维度。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <tr>
      <td>out</td>
      <td>输出</td>
      <td>函数的输出是输入的梯度值。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
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
| 图模式 | - | 通过[算子IR](op_graph/softmax_grad_proto.h)构图方式调用SoftmaxGrad算子。 |
