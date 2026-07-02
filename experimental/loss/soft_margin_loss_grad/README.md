# SoftMarginLossGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品|√|

## 功能说明

- 算子功能：求soft_margin_loss函数的梯度。

- 计算公式：

$$
grad\_input = -target \times \frac{exp(-target \times self)}{1 + exp(-target \times self)} \times grad\_output \times cof
$$

其中：

- 当 `reduction="mean"` 时，`cof = 1.0 / (shape_size)`
- 当 `reduction="none"` 时，`cof = 1.0`

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
      <td>前向传播的预测值输入。公式中的self</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>目标标签输入。公式中的target</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_out</td>
      <td>输入</td>
      <td>反向传播的梯度输入。公式中的grad_output</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>reduction</td>
      <td>属性</td>
      <td>规约模式，可选"none"或"mean"或"sum"。</td>
      <td>string</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>梯度计算结果输出。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

1. 输入 `x`、`y`、`grad_out` 及输出 `out` 的数据类型必须一致。
2. `x`、`y`、`grad_out` 的 shape 需要满足 broadcast 关系，且 `out` 的 shape 需要与三者广播后的结果一致。
3. `reduction` 参数支持以下取值：
   - `"none"`：不做规约，直接按元素计算梯度。
   - `"mean"`：对输出结果做均值规约。
   - `"sum"`：对输出结果做求和规约。
4. 当前默认 `reduction="mean"`。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_soft_margin_loss_grad.cpp](./examples/test_aclnn_soft_margin_loss_grad.cpp) | 通过[aclnnSoftMarginLossBackward](./docs/aclnnSoftMarginLossBackward.md)接口方式调用SoftMarginLossGrad算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| hth810 | 个人开发者 | SoftMarginLossGrad | 2026/3/19 | SoftMarginLossGrad算子适配开源仓 |
