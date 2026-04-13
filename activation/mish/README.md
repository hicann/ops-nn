# Mish

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    ×     |
| <term>Atlas 推理系列产品</term>                       |    ×     |
| <term>Atlas 训练系列产品</term>                       |    √     |

## 功能说明

- 算子功能：一个自正则化的非单调神经网络激活函数。

- 计算公式：

$$out_i = self_i \times \tanh(\mathrm{Softplus}(\mathrm{self}_i))$$

其中，$\mathrm{Softplus}(self_i) = \ln{(1 + \exp{(self_i)})}$

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
      <td>self</td>
      <td>输入</td>
      <td>待进行Mish激活计算的入参，对应计算公式中的self。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>Mish激活计算的出参，对应计算公式中的out。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16。

## 约束说明

- 确定性计算：aclnnMish&aclnnInplaceMish默认确定性实现。
- 输入shape限制：输入维度不能超过8维，仅支持0-8维。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_mish](./examples/test_aclnn_mish.cpp) | 通过[aclnnMish](./docs/aclnnMish&aclnnInplaceMish.md)接口方式调用Mish算子。 |
