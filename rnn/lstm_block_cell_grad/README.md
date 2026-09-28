# LSTMBlockCellGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     ×    |
|  <term>Atlas A2系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

LSTMBlockCellGrad计算TensorFlow LSTMBlockCell单元（含窥孔连接）的反向梯度：由当前时间步的cs_grad与h_grad，结合前向门控中间量i/cs/f/o/ci/co，重建上一时刻细胞状态梯度cs_prev_grad、融合门pre-activation梯度dicfo（列块[di, dc, df, do]，与前向融合权重w的icfo列序一致）以及三个窥孔权重梯度wci_grad/wcf_grad/wco_grad。语义与TensorFlow tf.raw_ops.LSTMBlockCellGrad完全对齐（16输入/5输出/use_peephole属性逐项一致），通过GE图模式调用，由tf_plugin认领TF前端构图节点。

use_peephole=false时三个窥孔权重梯度输出精确全0（与TF原生行为一致）。x/h_prev/w/b不进入梯度公式——它们的梯度由调用方通过dicfo在算子外重建（矩阵乘），本算子仅参与shape契约校验。

计算公式（符号约定：$\odot$ 为逐元素乘；$\sum_{b=0}^{B-1}$ 为沿batch维（axis 0）求和；$i$ / $f$ / $o$ 为sigmoid激活、$ci$ / $co$ 为tanh激活；$B$ = batch、$C$ = cell、$N$ = num_inputs）：

$$
dig = 1 - co \odot co, \qquad do_{pre} = h_{grad} \odot co \odot o \odot (1 - o)
$$

$$
dcs = cs_{grad} + h_{grad} \odot o \odot dig + \begin{cases} do_{pre} \odot wco, & \text{use\_peephole} = \text{true} \\ 0, & \text{use\_peephole} = \text{false} \end{cases}
$$

$$
di_{pre} = dcs \odot ci \odot i \odot (1 - i), \quad dci_{pre} = dcs \odot i \odot (1 - ci \odot ci), \quad df_{pre} = dcs \odot cs_{prev} \odot f \odot (1 - f)
$$

$$
cs_{prev\_grad} = dcs \odot f + \begin{cases} wci \odot di_{pre} + wcf \odot df_{pre}, & \text{use\_peephole} = \text{true} \\ 0, & \text{use\_peephole} = \text{false} \end{cases}
$$

$$
dicfo = \big[\, di_{pre} \;\big|\; dci_{pre} \;\big|\; df_{pre} \;\big|\; do_{pre} \,\big]
$$

$$
wci_{grad} = \begin{cases} \sum_{b=0}^{B-1} cs_{prev}[b,:] \odot di_{pre}[b,:], & \text{true} \\ \mathbf{0}, & \text{false} \end{cases} \quad wcf_{grad} = \begin{cases} \sum_{b=0}^{B-1} cs_{prev}[b,:] \odot df_{pre}[b,:], & \text{true} \\ \mathbf{0}, & \text{false} \end{cases} \quad wco_{grad} = \begin{cases} \sum_{b=0}^{B-1} cs[b,:] \odot do_{pre}[b,:], & \text{true} \\ \mathbf{0}, & \text{false} \end{cases}
$$

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 420px">
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
<tr><td>x</td><td>输入</td><td>细胞输入张量，shape为(batch, num_inputs)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>cs_prev</td><td>输入</td><td>上一时刻细胞状态，shape为(batch, cell)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>h_prev</td><td>输入</td><td>上一时刻输出，shape为(batch, cell)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>w</td><td>输入</td><td>权重矩阵，shape为(num_inputs+cell, 4*cell)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>wci</td><td>输入</td><td>输入门窥孔权重，shape为(cell,)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>wcf</td><td>输入</td><td>遗忘门窥孔权重，shape为(cell,)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>wco</td><td>输入</td><td>输出门窥孔权重，shape为(cell,)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>b</td><td>输入</td><td>偏置向量，shape为(4*cell,)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>i</td><td>输入</td><td>前向输入门激活值，shape为(batch, cell)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>cs</td><td>输入</td><td>前向tanh前的细胞状态，shape为(batch, cell)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>f</td><td>输入</td><td>前向遗忘门激活值，shape为(batch, cell)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>o</td><td>输入</td><td>前向输出门激活值，shape为(batch, cell)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>ci</td><td>输入</td><td>前向细胞门激活值，shape为(batch, cell)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>co</td><td>输入</td><td>前向tanh后的输出，shape为(batch, cell)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>cs_grad</td><td>输入</td><td>当前时刻cs的梯度，shape为(batch, cell)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>h_grad</td><td>输入</td><td>当前时刻h的梯度，shape为(batch, cell)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>use_peephole</td><td>属性</td><td>是否使用对角窥孔连接，bool类型，默认false；为false时三个窥孔权重梯度输出精确全0。</td><td>BOOL</td><td>-</td></tr>
<tr><td>cs_prev_grad</td><td>输出</td><td>cs_prev的梯度，shape为(batch, cell)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>dicfo</td><td>输出</td><td>四门梯度di、dc、df、do的拼接，shape为(batch, 4*cell)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>wci_grad</td><td>输出</td><td>输入门窥孔权重梯度，shape为(cell,)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>wcf_grad</td><td>输出</td><td>遗忘门窥孔权重梯度，shape为(cell,)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
<tr><td>wco_grad</td><td>输出</td><td>输出门窥孔权重梯度，shape为(cell,)。</td><td>FLOAT、FLOAT16</td><td>ND</td></tr>
</tbody>
</table>

## 约束说明

- 16个张量输入必须共享同一数据类型（FLOAT或FLOAT16），混合dtype组合在GEIR执行阶段被拒绝。
- batch维度必须一致（x为batch×num_inputs，其余二维输入为batch×cell）；w.shape必须为(num_inputs+cell, 4*cell)；b.shape必须为4*cell；wci/wcf/wco.shape必须为cell。
- 仅支持ND格式。
- 空张量（batch=0或cell=0）为合法输入，输出为对应空shape。
- use_peephole取值由调用方保证与TF前端构图一致（算子侧不校验该属性值域）。
- 确定性说明：默认确定性实现，同shape同输入多次执行结果位级一致。

## 调用说明

<table style="table-layout: fixed; width: 1000px">
<colgroup>
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 600px">
</colgroup>
<thead>
<tr>
<th>调用方式</th>
<th>样例代码</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td>图模式调用</td>
<td><a href="./examples/test_geir_lstm_block_cell_grad.cpp">test_geir_lstm_block_cell_grad</a></td>
<td>通过算子IR构图方式调用LSTMBlockCellGrad算子。</td>
</tr>
</tbody>
</table>
