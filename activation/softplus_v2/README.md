# SoftplusV2

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：Softplus激活函数，当$\beta \cdot x$不超过阈值时输出$\frac{1}{\beta}\log(1+\exp(\beta \cdot x))$，否则直接输出$x$。
- 计算公式：

$$
out = \begin{cases}
\displaystyle \frac{1}{\beta}\log(1+\exp(\beta \cdot x)), & \beta \cdot x \le threshold \\
x, & \beta \cdot x > threshold
\end{cases}
$$

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
<tr>
<th>参数名</th>
<th>输入/输出/属性</th>
<th>描述</th>
<th>数据类型</th>
<th>数据格式</th>
</tr>
</thead>
<tbody>
<tr>
<td>x</td>
<td>输入</td>
<td>待进行Softplus计算的输入张量，公式中的x。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>输出</td>
<td>Softplus计算后的输出张量，公式中的out，shape与x一致。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>beta</td>
<td>可选属性</td>
<td>公式中的beta，默认值为1.0。</td>
<td>FLOAT</td>
<td>-</td>
</tr>
<tr>
<td>threshold</td>
<td>可选属性</td>
<td>公式中的threshold，默认值为20.0。</td>
<td>FLOAT</td>
<td>-</td>
</tr>
</tbody>
</table>

- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| aclnn接口 | [test_aclnn_softplus](./examples/test_aclnn_softplus.cpp) | 通过aclnnSoftplusGetWorkspaceSize、aclnnSoftplus接口调用SoftplusV2算子，接口说明参见[aclnnSoftplus](./docs/aclnnSoftplus.md)。 |
