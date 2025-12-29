# MseLossGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|<term>Ascend 950PR/Ascend 950DT</term>   |     √    |

## 功能说明

- 算子功能：均方误差函数[aclnnMseLoss](aclnnMseLoss.md)的反向传播。

- 计算公式：

  当`reduction`为`mean`时：

  $$
  MselossBackward(grad, x, y) = grad * (x - y) * 2 / x.numel()
  $$

  其中`x.numel()`表示`x`中的元素个数。如果`reduction`不是`mean`, 那么：

  $$
  MselossBackward(grad, x, y) = grad * (x - y) * 2
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
      <td>predict</td>
      <td>输入</td>
      <td>公式中的输入grad 
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>输入</td>
      <td>公式中的输入x。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dout</td>
      <td>输入</td>
      <td>公式中的输入y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>输入</td>
      <td>公式中的输入reduction，指定损失函数的计算方式，支持 0('none') | 1('mean') | 2('sum')。'none' 表示不应用减少，'mean' 表示输出的总和将除以self中的元素数，'sum' 表示输出将被求和。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出MselossBackward。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无  

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_mse_loss_grad](./examples/test_aclnn_mse_loss_grad.cpp) | 通过[aclnnMseLossGrad](./docs/aclnnMseLossBackward.md)接口方式调用mse_loss_grad算子。    |