# SigmoidCrossEntropyWithLogitsGradV2

## 产品支持情况

|产品|是否支持|
|:---|:---:|
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|√|

## 功能说明

- 算子功能：计算二分类交叉熵（带 logits）损失对输入 logits 的梯度，支持可选输入 `weight` 和 `pos_weight`。
- 该算子对应接口名为 `aclnnBinaryCrossEntropyWithLogitsBackward`。

设输入为 `predict`（logits）、`target`、`dout`，定义：

$$
p = sigmoid(predict) = \frac{1}{1 + e^{-predict}}
$$

当 `pos_weight` 不存在时：

$$
grad\_base = p - target
$$

当 `pos_weight` 存在时：

$$
log\_weight = pos\_weight \cdot target
$$

$$
grad\_base = (log\_weight + 1 - target) \cdot p - log\_weight
$$

梯度输出为：

$$
gradient = grad\_base \cdot dout
$$

若 `weight` 非空：

$$
gradient = gradient \cdot weight
$$

当 `reduction = mean` 时，再执行：

$$
gradient = gradient \cdot \frac{1}{N}
$$

其中 $N$ 为元素总数。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
	<col style="width: 190px">
	<col style="width: 170px">
	<col style="width: 360px">
	<col style="width: 220px">
	<col style="width: 110px">
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
			<td>模型输出 logits。</td>
			<td>BFLOAT16、FLOAT16、FLOAT</td>
			<td>ND</td>
		</tr>
		<tr>
			<td>target</td>
			<td>输入</td>
			<td>标签张量。</td>
			<td>BFLOAT16、FLOAT16、FLOAT</td>
			<td>ND</td>
		</tr>
		<tr>
			<td>dout</td>
			<td>输入</td>
			<td>上游梯度输入。</td>
			<td>BFLOAT16、FLOAT16、FLOAT</td>
			<td>ND</td>
		</tr>
		<tr>
			<td>weight</td>
			<td>可选输入</td>
			<td>样本权重。</td>
			<td>BFLOAT16、FLOAT16、FLOAT</td>
			<td>ND</td>
		</tr>
		<tr>
			<td>pos_weight</td>
			<td>可选输入</td>
			<td>正样本权重。</td>
			<td>BFLOAT16、FLOAT16、FLOAT</td>
			<td>ND</td>
		</tr>
		<tr>
			<td>reduction</td>
			<td>可选属性</td>
			<td>支持 none、mean、sum。</td>
			<td>STRING / INT64（接口映射）</td>
			<td>-</td>
		</tr>
		<tr>
			<td>gradient</td>
			<td>输出</td>
			<td>梯度输出。</td>
			<td>BFLOAT16、FLOAT16、FLOAT</td>
			<td>ND</td>
		</tr>
	</tbody></table>

## 约束说明

- 输入维度范围 1-8 维。
- `predict`、`target`、`dout` 的 shape 需一致。
- `weight`、`pos_weight` 为可选输入，若存在需与 `predict` shape 一致。
- 输入与输出数据类型需保持一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:---|:---|:---|
| aclnn调用 | [test_aclnn_sigmoid_cross_entropy_with_logits_grad_v2](./examples/test_aclnn_sigmoid_cross_entropy_with_logits_grad_v2.cpp) | 通过 [aclnnSigmoidCrossEntropyWithLogitsGradV2](./docs/aclnnSigmoidCrossEntropyWithLogitsGradV2.md) 文档说明的两段式接口调用该算子。 |
