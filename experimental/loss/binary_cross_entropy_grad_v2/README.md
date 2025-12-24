# BinaryCrossEntropyGrad

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 功能说明

- 算子功能：计算二元交叉熵损失函数对输入的梯度，用于反向传播

- 计算公式：
给定样本真实标签 (logits)、模型预测概率 (labels)、上游梯度 (grad) 及可选权重 (weight)，算子逐元素计算关于预测值的梯度 (Z)：
$$
z_i= weight_i * grad_i * （logits_i - labels_i）
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"> <colgroup> 
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
    </tr> </thead> 
  <tbody> 
    <tr> 
      <td>grad</td> 
      <td>输入</td> 
      <td>上游梯度输入，公式中的grad_i。</td> 
      <td>FLOAT、FLOAT16、INT32、INT16</td> 
      <td>ND</td> 
    </tr> 
    <tr> 
      <td>logits</td> 
      <td>输入</td> 
      <td>模型预测的logits值，公式中的logits_i。</td> 
      <td>FLOAT、FLOAT16、INT32、INT16</td> 
      <td>ND</td> 
    </tr> 
    <tr> 
      <td>labels</td> 
      <td>输入</td> 
      <td>真实标签值，公式中的labels_i。</td> 
      <td>FLOAT、FLOAT16、INT32、INT16</td> 
      <td>ND</td> 
    </tr> 
    <tr> 
      <td>weight</td> 
      <td>输入</td> 
      <td>样本权重，公式中的weight_i。</td> 
      <td>FLOAT、FLOAT16、INT32、INT16</td> 
      <td>ND</td> 
    </tr> 
    <tr> 
      <td>z</td> 
      <td>输出</td> 
      <td>计算得到的梯度输出。</td> 
      <td>FLOAT、FLOAT16、INT32、INT16</td> 
      <td>ND</td> 
    </tr> 
    <tr> 
      <td>reduction</td> 
      <td>属性</td> 
      <td>归约模式：0=none（无归约），1=sum（求和），2=mean（平均）。</td> 
      <td>INT</td> 
      <td>-</td> 
    </tr> 
    </tbody> 
    </table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_binary_cross_entropy_grad_v2.cpp](./examples/test_aclnn_binary_cross_entropy_grad_v2.cpp) | 通过[test_aclnn_binary_cross_entropy_grad_v2]接口方式调用BinaryCrossEntropyGrad算子。 |


