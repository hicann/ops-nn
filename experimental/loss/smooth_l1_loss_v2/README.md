# SmoothL1LossV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：平滑 L1 损失计算，支持 `none`、`mean`、`sum` 三种 reduction 模式。

- 计算公式：

  设输入为 `predict` 与 `label`，定义：

  $$
  diff = predict - label
  $$

  $$
  abs\_diff = |diff|
  $$

  $$
  min\_part = \min(abs\_diff, \sigma)
  $$

  $$
  loss = \frac{min\_part^2}{2\sigma} + (abs\_diff - min\_part)
  $$

  等价分段形式为：

  $$
  loss =
  \begin{cases}
  \frac{(predict-label)^2}{2\sigma}, & |predict-label| < \sigma \\
  |predict-label| - 0.5\sigma, & |predict-label| \ge \sigma
  \end{cases}
  $$

  reduction 说明：

	- `none`：返回逐元素损失。
	- `sum`：返回所有元素损失和。
	- `mean`：返回所有元素损失均值。

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
			<td>预测值张量，公式中的 predict。</td>
			<td>BFLOAT16、FLOAT16、FLOAT</td>
			<td>ND</td>
		</tr>
		<tr>
			<td>label</td>
			<td>输入</td>
			<td>标签张量，公式中的 label。</td>
			<td>BFLOAT16、FLOAT16、FLOAT</td>
			<td>ND</td>
		</tr>
		<tr>
			<td>sigma</td>
			<td>可选属性</td>
			<td>平滑阈值，控制二次段与一次段分界，默认值为 1.0。aclnn 接口参数名为 beta，与算子属性 sigma 语义一致。</td>
			<td>FLOAT</td>
			<td>ND</td>
		</tr>
		<tr>
			<td>reduction</td>
			<td>可选属性</td>
			<td>支持 none、mean、sum。none 返回逐元素损失；mean/sum 返回标量。</td>
			<td>STRING / INT64（接口映射）</td>
			<td>ND</td>
		</tr>
		 <tr>
			<td>loss</td>
			<td>输出</td>
			<td>公式中的输出 loss。</td>
			<td>BFLOAT16、FLOAT16、FLOAT</td>
			<td>ND</td>
		</tr>
	</tbody></table>

## 约束说明

- 输入维度范围 1-8 维。
- `predict` 与 `label` 的 dtype 需要一致。
- `predict` 与 `label` 不支持广播；二者的 shape 需要完全一致。当 `reduction=none` 时，输出 shape 与输入 shape 一致。
- `sigma` 必须为正数。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_smooth_l1_loss_v2](./examples/test_aclnn_smooth_l1_loss_v2.cpp) | 通过[aclnnSmoothL1Loss](./docs/aclnnsmoothl1lossv2.md)文档说明的两段式接口调用该算子。    |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| WO | 浙江工业大学-智能计算研究所 | SmoothL1LossV2 | 2026/04/21 | SmoothL1LossV2算子适配开源仓 |
