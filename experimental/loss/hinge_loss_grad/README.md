# HingeLossGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend910B4|?|

## 功能说明

- 算子功能：求hinge_loss函数的梯度。

- 计算公式：

$$
grad\_input = (1 - target \times predict > 0) \ ? (-target \times grad\_output) : 0
$$

其中：
- `predict`：模型预测值
- `target`：真实标签，通常取值为 +1 或 -1
- `grad_output`：来自上游的反向传播梯度
- 当 $1 - y \cdot f(x) > 0$ 时，梯度为 $-y \cdot grad\_output$；否则为 0

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
      <td>predict</td>
      <td>输入</td>
      <td>前向传播的预测值输入。</td>
      <td>fp32/fp16/bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td>真实标签输入，通常取值为+1或-1。</td>
      <td>fp32/fp16/bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_output</td>
      <td>输入</td>
      <td>反向传播的梯度输入。</td>
      <td>fp32/fp16/bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>梯度计算结果输出，shape与predict一致。</td>
      <td>fp32/fp16/bf16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

1. 输入 `predict`、`target`、`grad_output` 及输出 `grad_input` 的数据类型必须一致，支持 fp32、fp16、bf16。
2. 输入 `predict`、`target`、`grad_output` 的 shape 必须完全相同。
3. 输出 `grad_input` 的 shape 与输入 shape 一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_hinge_loss_grad.cpp](./examples/test_aclnn_hinge_loss_grad.cpp) | 通过[aclnnHingeLossGrad](./docs/aclnnHingeLossGrad.md)接口方式调用HingeLossGrad算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Shi Xiangyang (shi-xiangyang225) | AISS Group, Harbin Institute of Technology (HIT) | HingeLossGrad | 2026/5/11 | HingeLossGrad算子适配开源仓（多核tiling、aclnn测试通过） |
