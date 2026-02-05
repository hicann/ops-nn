# SmoothL1LossV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：计算SmoothL1损失函数。
- 计算公式:
  
  Batch为N的损失函数，当`reduction`为none时，此函数定义为：

  $$
  \ell(x,y) = L = \{l_1,\dots,l_N\}^\top
  $$
  
  其中的$l_n$为：
  
  $$
  l_n = \begin{cases}
  0.5(x_n-y_n)^2/beta, & if |x_n-y_n| < beta \\
  |x_n-y_n| - 0.5*beta, &  otherwise
  \end{cases}
  $$

  如果`reduction`为`mean`或`sum`时，

  $$
  \ell(x,y)=\begin{cases}
  mean(L), & \text{if reduction} = \text{mean}\\
  sum(L), & \text{if reduction} = \text{sum}
  \end{cases}
  $$
- 其中：
  - Xn: predictions
  - Yn: labels/truth


## 参数说明

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>predict</td>
      <td>输入</td>
      <td>公式中的'x'，表示用于计算SmoothL1损失的预测值输入。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>输入</td>
      <td>公式中的'y'，表示predict需要拟合的真实标签值输入。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示predict与label经smooth_l1_loss_v2计算得到的损失值输出</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>属性</td>
      <td>控制smooth_l1_loss_v2中MSE与L1损失分界边界的阈值参数</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>属性</td>
      <td>表示对逐元素损失值进行均值/求和/无归约的归约参数</td>
      <td>STRING</td>
      <td>-</td>
    </tr>

  </tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_smooth_l1_loss_v2.cpp](examples/test_aclnn_smooth_l1_loss_v2.cpp) | 通过[aclnnsmoothL1Loss](docs/aclnnSmoothL1Loss.md)接口方式调用SmoothL1Loss算子。 |