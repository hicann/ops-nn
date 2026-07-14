# HuberLossGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|×|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|×|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|
|Atlas 200I/500 A2 推理产品|×|
|Atlas 推理系列产品|×|
|Atlas 训练系列产品|×|

## 功能说明

- 算子功能：完成 HuberLoss 梯度计算。

- 计算公式：

$$
grad\_output =
\begin{cases}
predictions - targets, & |predictions - targets| \leq \delta \\
\delta \cdot sign(predictions - targets), & |predictions - targets| > \delta
\end{cases}
$$

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
      <td>predictions</td>
      <td>输入</td>
      <td>预测值，公式中的 predictions。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>targets</td>
      <td>输入</td>
      <td>目标值，公式中的 targets。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>delta</td>
      <td>属性</td>
      <td>HuberLoss 阈值参数。默认值为 1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_output</td>
      <td>输出</td>
      <td>HuberLoss 梯度计算结果。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="./examples/test_aclnn_huber_loss_grad.cpp">test_aclnn_huber_loss_grad</a></td>
    <td>参见算子调用完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
