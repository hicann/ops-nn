# ModulateGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：完成Modulate反向传播中参数的计算，进行梯度更新。
- 计算公式：
    
    设输入self的shape为[B, L, D]计算公式如下：
    公式：

    $$
    \begin{cases}
    \text{grad\_x} = \text{grad\_output} \odot \left(1 + \text{scale}^{\uparrow L}\right) \\
    \text{grad\_scale} = \sum_{l=1}^{L} (\text{grad\_output} \odot \text{input})_{b,l,d} \\
    \text{grad\_shift} = \sum_{l=1}^{L} \text{grad\_output}_{b,l,d}
    \end{cases}
    $$

    符号说明：
    - $\odot$: 表示逐元素乘法；
    - $\sum_{l=1}^{L}$: 求和操作，沿序列维度$L$(即dim=1)进行
    -  $b,l,d$：下标，表示张量的维度索引（通常为Batch，Length，Dimension） 
    - $\text{scale}^{\uparrow L}$： 表示将scale张量在序列维度 $L$ 上进行广播（扩展）

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>grad_output</td>
      <td>输入</td>
      <td>表示传入的特征张量，公式中的grad_output。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>表示传入的特征张量，公式中的input。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>可选参数，表示缩放的系数，公式中的scale。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>输入</td>
      <td>可选参数，表示平移的系数，公式中的shift。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>表示缩放后的输出张量，公式中的grad_input。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
        <tr>
      <td>grad_scale</td>
      <td>输出</td>
      <td>表示缩放后的输出张量，公式中的grad_scale。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
        <tr>
      <td>grad_shift</td>
      <td>输出</td>
      <td>表示缩放后的输出张量，公式中的grad_shift。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_modulatebackward](./examples/test_aclnn_modulatebackward.cpp) | 通过[aclnnModulateBackward](./docs/aclnnModulate.md)接口方式调用ModulateGrad算子。    |


