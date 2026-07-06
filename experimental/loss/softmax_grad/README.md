# SoftmaxGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品|√|

## 功能说明

- 算子功能：求softmax函数的梯度。

- 计算公式：

$$
dot_i = \sum_j (dy_{i,j} \cdot y_{i,j})
$$
$$
dx_{i,j} = y_{i,j} \cdot (dy_{i,j} - dot_i)
$$

其中：
- `y`：softmax正向输出（每行概率和为1），shape [N, C]
- `dy`：上游梯度，shape [N, C]
- `dx`：对softmax输入logits的梯度，shape [N, C]

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
      <td>y</td>
      <td>输入</td>
      <td>softmax正向输出，每行概率和为1。</td>
      <td>fp32/fp16/bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>上游梯度输入。</td>
      <td>fp32/fp16/bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>梯度计算结果输出，shape与y一致。</td>
      <td>fp32/fp16/bf16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

1. 输入 `y`、`dy` 及输出 `dx` 的数据类型必须一致，支持 fp32、fp16、bf16。
2. 输入 `y` 和 `dy` 的 shape 必须完全相同。
3. 输出 `dx` 的 shape 与输入 shape 一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_softmax_grad.cpp](./examples/test_aclnn_softmax_grad.cpp) | 通过[aclnnSoftmaxGrad](./docs/aclnnSoftmaxGrad.md)接口方式调用SoftmaxGrad算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Shi Xiangyang (shi-xiangyang225) | AISS Group, Harbin Institute of Technology (HIT) | SoftmaxGrad | 2026/5/13 | SoftmaxGrad算子适配开源仓（多核tiling、aclnn测试通过） |
