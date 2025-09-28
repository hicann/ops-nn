# MseLossV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：
  
  计算输入x和目标y中每个元素之间的均方误差。

- 计算公式：
  
  当`reduction`为`none`时：

$$
  \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
  l_n = \left( x_n - y_n \right)^2,
$$

  其中$x$是self，$y$是target，$N$是batch的大小。如果`reduction`不是`none`, 那么：

$$
  \ell(x, y) =
  \begin{cases}
      \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
      \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
  \end{cases}
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>公式中的输入x 
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td>公式中的输入y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>输入</td>
      <td>公式中的输入reduction，指定要应用到输出的缩减，支持 0('none') | 1('mean') | 2('sum')。'none' 表示不应用缩减，'mean' 表示输出的总和将除以输出中的元素数，'sum' 表示输出将被求和。</td>
      <td>int64_t</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出$\ell(x, y)$，当reduction的值为0时，out与self、target做broadcast后的tensor的shape一致；当reduction的值为1或2时，out是0维tensor。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无  

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_mse_loss_v2](./examples/test_aclnn_mse_loss_v2.cpp) | 通过[aclnnMseLoss](../mse_loss/docs/aclnnMseLoss.md)接口方式调用mse_loss_v2算子。    |
