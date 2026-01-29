# Softplus

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>                   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>   |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>                   |     ×    |
|  <term>Atlas 推理系列产品</term>                           |     √    |
|  <term>Atlas 训练系列产品</term>                           |     √    |

## 功能说明

- 算子功能：激活函数，用于将任意实数映射到正数区间。

- 计算公式：

  $$
  y = log(1 + e^x)
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1000px">
<colgroup>
<col style="width: 150px">
<col style="width: 150px">
<col style="width: 300px">
<col style="width: 250px">
<col style="width: 150px">
</colgroup>
<thead>
	<tr>
		<th>参数名</th>
		<th>输入/输出</th>
		<th>描述</th>
		<th>数据类型</th>
		<th>数据格式</th>
		</tr>
	</thead>
	<tbody>
		<tr>
		<td>x</td>
		<td>输入</td>
		<td>表示softplus激活函数的输入，公式中的x。</td>
		<td>FLOAT、FLOAT16、BFLOAT16</td>
		<td>ND</td>
		</tr>
		<tr>
		<td>y</td>
		<td>输出</td>
		<td>表示softplus激活函数的输出，公式中的y。</td>
		<td>FLOAT、FLOAT16、BFLOAT16</td>
		<td>ND</td>
		</tr>
	</tbody>
</table>
  
## 约束说明

输入x和输出y的shape和数据类型需要一致

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_softplus.cpp](examples/test_geir_softplus.cpp) | 通过[算子IR](./op_graph/softplus_proto.h)构图方式调用softplus算子。 |