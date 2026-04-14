# SoftMarginLoss

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 功能说明

- 算子功能：计算二分类场景下的soft margin loss（逻辑回归损失），对输入预测值和目标值逐元素计算损失，并支持归约输出。

- 计算公式：
  给定预测张量$self$和目标张量$target$，逐元素计算损失：

  $
  L_i = \max(0, -t_i \cdot x_i) + \log(1 + \exp(-|t_i \cdot x_i|))
  $

  其中$x_i = self_i$，$t_i = target_i$。

  归约模式：
  - $reduction = 0$（none）：$output = L$，输出与输入同shape
  - $reduction = 1$（mean）：$output = \frac{1}{N}\sum_{i}L_i$，输出标量
  - $reduction = 2$（sum）：$output = \sum_{i}L_i$，输出标量

- 示例：
    假设输入张量$self = [1, -1, 2, -2]$，目标张量$target = [1, 1, -1, -1]$，$reduction = 0$（none），那么输出张量$output = [0.3133, 1.3133, 2.1269, 0.1269]$，具体计算过程如下：
    $
    \begin{aligned}
    L_0 &= \max(0, -(1)(1)) + \log(1 + \exp(-|1|)) = 0 + \log(1 + 0.3679) = 0.3133 \\
    L_1 &= \max(0, -(1)(-1)) + \log(1 + \exp(-|-1|)) = 1 + \log(1 + 0.3679) = 1.3133 \\
    L_2 &= \max(0, -(-1)(2)) + \log(1 + \exp(-|-2|)) = 2 + \log(1 + 0.1353) = 2.1269 \\
    L_3 &= \max(0, -(-1)(-2)) + \log(1 + \exp(-|-2|)) = 0 + \log(1 + 0.1353) = 0.1269
    \end{aligned}
    $

## 参数说明

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 60px">
  <col style="width: 60px">
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 60px">
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
      <td>预测值张量，公式中的x。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td>目标值张量，shape和dtype与self一致，公式中的t。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>输出张量。reduction=none时与输入同shape；reduction=mean/sum时为标量（0维张量）。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>属性</td>
      <td><ul><li>归约模式：0=none，1=mean，2=sum。</li><li>默认值为1（mean）。</li></ul></td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- self和target的shape和dtype必须一致。
- float16输入在Kernel内部提升至float32计算，仅最终输出转回float16。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_soft_margin_loss.cpp](./examples/test_aclnn_soft_margin_loss.cpp) | 通过aclnn接口方式调用SoftMarginLoss算子。 |
