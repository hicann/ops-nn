# PReluGradReduce

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
| <term>Atlas 推理系列产品</term>    |     ×    |
| <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：激活函数，返回与输入tensor shape相同的tensor，tensor中value大于等于0时，取该value，小于0，取0。
- 计算公式：

$$
  gradInput
  =
  \begin{cases}
  gradOutput, & self > 0 \\
  weight \times gradOutput, & self \le 0
  \end{cases}
  $$

  $$
  update
  =
  \begin{cases}
  0, & self > 0 \\
  self \times gradOutput, & self \le 0
  \end{cases}
  $$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1305px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 247px">
  <col style="width: 108px">
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
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>grads</td>
      <td>输入</td>
      <td>prelu正向的输出y反向传播的梯度</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
  <tr>
      <td>features</td>
      <td>输入</td>
      <td>prelu正向的输入features</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
  <tr>
      <td>weights</td>
      <td>输入</td>
      <td>prelu正向的输入weights</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>

  <tr>
      <td>updates</td>
      <td>输入</td>
      <td>weights梯度计算的中间结果，用于计算da</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
  <tr>
      <td>da</td>
      <td>输出</td>
      <td>weights的梯度</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
  </table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_prelu_backward](../p_relu_grad_update/examples/test_aclnn_prelu_backward.cpp) | 通过[aclnnPReluBackward](../p_relu_grad_update/docs/aclnnPreluBackward.md)接口方式调用PReluGradReduce算子。 |
| 图模式 | - | 通过[算子IR]()构图方式调用PReluGradReduce算子。 |