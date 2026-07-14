# ReluGradV3

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品|√|

## 功能说明

- 算子功能：求ReLU函数的梯度。

- 计算公式：

$$
z = (x > 0) \ ? \ y : 0
$$

其中：
- `x`：前向ReLU的输入
- `y`：来自上游的反向传播梯度（grad_output）
- `z`：梯度计算结果（grad_input），当 x > 0 时等于 y，否则为 0

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
      <td>前向ReLU的输入。</td>
      <td>fp32/fp16/bf16/int32/uint8/int8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>反向传播的梯度输入（grad_output）。</td>
      <td>fp32/fp16/bf16/int32/uint8/int8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>z</td>
      <td>输出</td>
      <td>梯度计算结果输出（grad_input），shape与x一致。</td>
      <td>fp32/fp16/bf16/int32/uint8/int8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

1. 输入 `x`、`y` 及输出 `z` 的数据类型必须一致，支持 fp32、fp16、bf16。
2. 输入 `y` 支持与 `x` shape 完全相同，或作为单元素标量广播到 `x` 的 shape。
3. 输出 `z` 的 shape 与输入 `x` 一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_relu_grad_v3.cpp](./examples/test_aclnn_relu_grad_v3.cpp) | 通过[aclnnReluGradV3](./docs/aclnnReluGradV3.md)接口方式调用ReluGradV3算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Shi Xiangyang (shi-xiangyang225) | AISS Group, Harbin Institute of Technology (HIT) | ReluGradV3 | 2026/5/13 | ReluGradV3算子适配开源仓（多核tiling、CompareScalar/Select API、aclnn测试通过） |
