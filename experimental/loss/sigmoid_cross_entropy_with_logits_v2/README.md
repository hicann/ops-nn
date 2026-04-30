# SigmoidCrossEntropyWithLogitsV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：二分类交叉熵（带 logits）损失计算，支持可选输入 weight 与 pos_weight。

- 计算公式：

  设输入为 self（logits）与 target，定义：

  $$
  max\_val = \max(-self, 0)
  $$

  $$
  log\_term = max\_val + \log\left(\exp(-max\_val) + \exp(-self - max\_val)\right)
  $$

  不带 posWeight 时：

  $$
  loss = (1 - target) * self + log\_term
  $$

  带 posWeight 时：

  $$
  log\_weight = (posWeight - 1) * target + 1
  $$

  $$
  loss = (1 - target) * self + log\_weight * log\_term
  $$

  若 weight 非空：

  $$
  loss = loss * weight
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
			<td>模型输出 logits，公式中的 predict。</td>
			<td>BFLOAT16、FLOAT16、FLOAT</td>
			<td>ND</td>
		</tr>
		<tr>
			<td>target</td>
			<td>输入</td>
			<td>标签张量，公式中的 target。</td>
			<td>BFLOAT16、FLOAT16、FLOAT</td>
			<td>ND</td>
		</tr>
		<tr>
			<td>weight</td>
			<td>可选输入</td>
			<td>样本权重，可广播到 predict 形状。</td>
			<td>BFLOAT16、FLOAT16、FLOAT</td>
			<td>ND</td>
		</tr>
		<tr>
			<td>pos_weight</td>
			<td>可选输入</td>
			<td>正样本权重，可广播到 predict 形状。</td>
			<td>BFLOAT16、FLOAT16、FLOAT</td>
			<td>ND</td>
		</tr>
		<tr>
			<td>reduction</td>
			<td>可选属性</td>
			<td>支持 none、mean、sum。none 返回逐元素损失；mean/sum 在上层接口中完成归约。</td>
			<td>STRING / INT64（接口映射）</td>
			<td>ND</td>
		</tr>
		 <tr>
			<td>loss</td>
			<td>输出</td>
			<td>公式中的输出 loss。</td>
			<td>FLOAT</td>
			<td>ND</td>
		</tr>
	</tbody></table>

## 约束说明

- 输入维度范围 1-8 维。
- predict 与 target 的 shape 需要一致。
- weight 与 pos_weight 为可选输入，若存在需可广播到 target 的 shape。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sigmoid_cross_entropy_with_logits_v2](./examples/test_aclnn_sigmoid_cross_entropy_with_logits_v2.cpp) | 通过[aclnnSigmoidCrossEntropyWithLogitsV2](./docs/aclnnSigmoidCrossEntropyWithLogitsV2.md)文档说明的两段式接口调用该算子。    |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| qianxun | 浙江工业大学-智能计算研究所 | SigmoidCrossEntropyWithLogitsV2 | 2026/04/16 | SigmoidCrossEntropyWithLogitsV2算子适配开源仓 |