# Erfinv

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

- 算子功能：erfinv是高斯误差函数erf的反函数。返回输入Tensor中每个元素对应在标准正态分布函数的分位数。

- 计算公式：

$$y=\mathrm{erfinv}(x)$$

其中 $x=erf(y)=\frac{2}{\sqrt{\pi}} \int_{0}^{y} e^{-t^2} dt$

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
      <td>x</td>
      <td>输入</td>
      <td>待进行erfinv计算的入参，对应计算公式中的x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>erfinv计算的出参，对应计算公式中的y。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas 训练系列产品：数据类型支持FLOAT、FLOAT16。

## 约束说明

- 确定性计算：aclnnErfinv&aclnnInplaceErfinv默认确定性实现。
- 输入shape限制：输入维度不能超过8维，仅支持0-8维。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_erfinv](./examples/test_aclnn_erfinv.cpp) | 通过[aclnnErfinv](./docs/aclnnErfinv&aclnnInplaceErfinv.md)接口方式调用Erfinv算子。 |
