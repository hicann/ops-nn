# MseLossGradV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     √    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：均方误差损失算子（[MseLossV2](../mse_loss_v2/README.md)）的反向传播，输出损失对预测值predict的梯度。

- 计算公式：

  当`reduction`为`mean`时：

  $$
  MselossBackward(grad, x, y) = grad * (x - y) * 2 / x.numel()
  $$

  其中`x.numel()`表示`x`中的元素个数。如果`reduction`不是`mean`，那么：

  $$
  MselossBackward(grad, x, y) = grad * (x - y) * 2
  $$

- 其中：
  - grad：上层回传的梯度（dout）。
  - x：前向的预测值（predict）。
  - y：前向的标签值（label）。

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
      <td>公式中的输入x，前向的预测值，需与label、dout满足broadcast关系。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>输入</td>
      <td>公式中的输入y，前向的标签值，需与predict、dout满足broadcast关系。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dout</td>
      <td>输入</td>
      <td>公式中的输入grad，上层回传的梯度，需与predict、label满足broadcast关系。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>属性</td>
      <td>指定损失函数的计算方式，支持none | mean | sum，默认mean。'none'表示不应用缩减，'mean'表示输出总和将除以predict的元素数，'sum'表示输出将被求和。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
     <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出MselossBackward，shape为predict、label、dout broadcast之后的结果，dtype与predict一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas推理系列产品</term>：数据类型不支持BFLOAT16。

## 约束说明

- 输入predict、label、dout的数据类型需保持一致，且必须满足broadcast关系；输出y的shape为三者broadcast之后的结果。
- 空tensor：predict、label、dout任一为空tensor时输出为空tensor，空进空出。
- 确定性计算：
  - aclnnMseLossBackward默认确定性实现。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_mse_loss_grad_v2](./examples/test_aclnn_mse_loss_grad_v2.cpp) | 通过[aclnnMseLossBackward](./docs/aclnnMseLossBackward.md)接口方式调用MseLossGradV2算子。 |

## 参考资源

- [aclnnMseLossBackward接口文档](./docs/aclnnMseLossBackward.md)
- [aclnnMseLoss前向接口文档](../mse_loss/docs/aclnnMseLoss.md)
- [aclnn调用样例](./examples/test_aclnn_mse_loss_grad_v2.cpp)
