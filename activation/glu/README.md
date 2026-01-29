# Glu

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 接口功能：GLU是一个门控线性单元函数，它将输入张量沿着指定的维度dim平均分成两个张量，并将其前部分张量与后部分张量的Sigmoid函数输出的结果逐元素相乘。
- 计算公式：

  $$
  GLU(a,b)=a \otimes \sigma(b)
  $$

  a表示的是输入张量根据指定dim进行均分后的前部分张量，b表示后半部分张量。

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
      <td>输入一个张量。</td>
      <td>DT_BF16、FLOAT16、DT_FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>待进行Glu计算的入参，将该输入张量沿着指定的维度dim平均分成公式中的两个张量a,b。</td>
      <td>DOUBLE、FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>

    </tr>
      <tr>
      <td>dim</td>
      <td>输出</td>
      <td>表示要拆分输入self的维度。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
     <tr>
      <td>out</td>
      <td>输出</td>
      <td>计算出参。</td>
      <td>DOUBLE、FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明
| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_glu](examples/test_aclnn_glu.cpp) | 通过[aclnnGlu](docs/aclnnGlu.md)接口方式调用glu算子。 |