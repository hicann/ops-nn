# ThnnFusedGruCell

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：对GRU（Gated Recurrent Unit）的单个时间步执行门控融合计算，输入两路门控预激活（输入侧与隐层侧）、上一步隐状态及可选的两路bias，一次计算产出新隐状态hy与反向计算复用的中间量storage，适用于循环神经网络逐步推理或训练中的高频单步计算。
- 计算公式：门控预激活沿最后一维按门序[r, z, n]三等分为gi_r、gi_z、gi_n与gh_r、gh_z、gh_n；两路bias同步三等分为b1_r、b1_z、b1_n与b2_r、b2_z、b2_n，并沿batch维广播；hx为上一步隐状态，B为batch维大小，H为隐藏维大小。

  $$
  rg = \frac{1}{1 + e^{-(gi_r + gh_r + b1_r + b2_r)}}
  $$

  $$
  zg = \frac{1}{1 + e^{-(gi_z + gh_z + b1_z + b2_z)}}
  $$

  $$
  ng = \tanh(gi_n + b1_n + rg \times (gh_n + b2_n))
  $$

  $$
  hy = ng + zg \times (hx - ng)
  $$

  $$
  storage = [rg \mid zg \mid ng \mid hx \mid (gh_n + b2_n)]
  $$

- 公式变量：hy为新隐状态；storage为反向复用中间量，五段[rg、zg、ng、hx、gh_n+b2_n]沿最后一维拼接，每段H列；input_bias与hidden_bias缺省时等价于全零bias。
- 精度说明：FLOAT16、BFLOAT16输入在float32中间精度下完成sigmoid、tanh与乘加计算，计算结果舍回原数据类型；输出数据类型与输入一致。

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
<td>input_gates</td>
<td>输入</td>
<td>输入侧门控预激活，对应公式中的gi_r、gi_z、gi_n，shape为(B, 3H)。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>hidden_gates</td>
<td>输入</td>
<td>隐层侧门控预激活，对应公式中的gh_r、gh_z、gh_n，shape为(B, 3H)。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>hx</td>
<td>输入</td>
<td>上一步隐状态，对应公式中的hx，shape为(B, H)。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>input_bias</td>
<td>可选输入</td>
<td>输入侧bias，对应公式中的b1_r、b1_z、b1_n，shape为(3H,)；缺省时等价于全零bias。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>hidden_bias</td>
<td>可选输入</td>
<td>隐层侧bias，对应公式中的b2_r、b2_z、b2_n，shape为(3H,)；缺省时等价于全零bias。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>hy</td>
<td>输出</td>
<td>新隐状态，对应公式中的hy，shape为(B, H)。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>storage</td>
<td>输出</td>
<td>反向复用中间量，对应公式中的storage，shape为(B, 5H)。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
</tbody>
</table>

## 约束说明

- 输入与输出的数据类型必须一致，仅支持BFLOAT16、FLOAT16、FLOAT；不支持跨数据类型组合，也不支持DOUBLE、INT64等其它数据类型。
- input_gates与hidden_gates的shape必须相同且为(B, 3H)，hx的shape为(B, H)，需满足input_gates.shape[1] == 3 × hx.shape[1]；input_bias与hidden_bias在位时元素个数必须为3H且两者相同。
- 输入与输出的数据格式仅支持ND。
- B=0或H=0（numel为0的空Tensor）为合法输入，直接返回空输出。
- aclnn接口中可选bias以空指针表达缺省，等价于全零bias；输入与输出均支持非连续Tensor（输入由接口层自动连续化，输出由接口层按声明的布局逐元素写回）。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn API | - | 通过[aclnnThnnFusedGruCell](docs/aclnnThnnFusedGruCell.md)接口调用，包含aclnnThnnFusedGruCellGetWorkspaceSize与aclnnThnnFusedGruCell两个接口的两段式调用。 |
| GE图模式 | - | 通过[算子IR定义](op_graph/thnn_fused_gru_cell_proto.h)构图调用。 |
