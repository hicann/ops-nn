# GRUBlockCellGrad

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

GRUBlockCellGrad是GRU block cell的单时间步反向梯度算子：输入上游梯度d_h与前向缓存的门控中间量r、u、c，计算对输入x的梯度d_x、对上一时刻隐状态h_prev的梯度d_h_prev，以及门预激活梯度拼接矩阵d_c_bar与d_r_bar_u_bar。算子在单个kernel内完成两条GEMM与逐元素梯度链的融合计算，服务于TensorFlow图模式下GRU网络的训练反向传播，与前向算子GRUBlockCell成对使用。

计算公式（∘为逐元素乘，×为矩阵乘，[a | b]为按列拼接，w_c^T与w_ru^T为权重矩阵的转置）：

$$
d\_c\_bar = d\_h \circ (1-u) \circ (1 - c \circ c)
$$

$$
d\_u\_bar = d\_h \circ (h\_prev - c) \circ u \circ (1-u)
$$

$$
[d\_x\_component\_2 \mid d\_h\_prevr] = d\_c\_bar \times w\_c^{T}
$$

$$
d\_r\_bar = (d\_h\_prevr \circ h\_prev \circ r) \circ (1-r)
$$

$$
d\_r\_bar\_u\_bar = [d\_r\_bar \mid d\_u\_bar]
$$

$$
[d\_x\_component\_1 \mid d\_h\_prev\_component\_1] = d\_r\_bar\_u\_bar \times w\_ru^{T}
$$

$$
d\_x = d\_x\_component\_1 + d\_x\_component\_2
$$

$$
d\_h\_prev = d\_h\_prev\_component\_1 + d\_h\_prevr \circ r + d\_h \circ u
$$

其中input_size为x第1维的长度，cell_size为h_prev第1维的长度；d_x_component_1、d_x_component_2取两次GEMM输出矩阵的前input_size列，d_h_prev_component_1、d_h_prevr取对应输出矩阵的后cell_size列。权重与偏置的梯度不在本算子内计算，由上层框架使用d_c_bar、d_r_bar_u_bar继续完成。

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
<td>当前时间步输入，其shape定义batch与input_size；数值不进入计算链，仅参与参数校验，对齐竞品接口。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>h_prev</td>
<td>输入</td>
<td>上一时刻隐状态，对应公式中h_prev，其shape定义batch与cell_size。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>w_ru</td>
<td>输入</td>
<td>重置门与更新门权重，对应公式中w_ru，shape为(input_size+cell_size, 2*cell_size)。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>w_c</td>
<td>输入</td>
<td>候选状态权重，对应公式中w_c，shape为(input_size+cell_size, cell_size)。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>b_ru</td>
<td>输入</td>
<td>重置门与更新门偏置，shape为(2*cell_size,)；数值不进入计算链，仅参与参数校验，对齐竞品接口。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>b_c</td>
<td>输入</td>
<td>候选状态偏置，shape为(cell_size,)；数值不进入计算链，仅参与参数校验，对齐竞品接口。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>r</td>
<td>输入</td>
<td>前向重置门输出，对应公式中r，shape为(batch, cell_size)。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>u</td>
<td>输入</td>
<td>前向更新门输出，对应公式中u，shape为(batch, cell_size)。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>c</td>
<td>输入</td>
<td>前向候选状态输出，对应公式中c，shape为(batch, cell_size)。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>d_h</td>
<td>输入</td>
<td>新隐状态的上游梯度，对应公式中d_h，shape为(batch, cell_size)。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>d_x</td>
<td>输出</td>
<td>对输入x的梯度，对应公式中d_x，shape为(batch, input_size)。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>d_h_prev</td>
<td>输出</td>
<td>对上一时刻隐状态的梯度，对应公式中d_h_prev，shape为(batch, cell_size)。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>d_c_bar</td>
<td>输出</td>
<td>对候选状态预激活的梯度，对应公式中d_c_bar，shape为(batch, cell_size)。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>d_r_bar_u_bar</td>
<td>输出</td>
<td>重置门与更新门预激活梯度的按列拼接，对应公式中d_r_bar_u_bar，shape为(batch, 2*cell_size)。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
</tbody>
</table>

## 约束说明

- 仅支持FLOAT数据类型与ND数据格式。
- 输入形状由(batch, input_size, cell_size)三元组唯一决定，batch、input_size取值需大于等于0，cell_size需大于0；各张量shape绑定关系为x==(batch, input_size)、h_prev==r==u==c==d_h==(batch, cell_size)、w_ru==(input_size+cell_size, 2*cell_size)、w_c==(input_size+cell_size, cell_size)、b_ru==(2*cell_size,)、b_c==(cell_size,)，输出d_x==(batch, input_size)、d_h_prev==d_c_bar==(batch, cell_size)、d_r_bar_u_bar==(batch, 2*cell_size)，rank或shape不匹配的参数组合会在校验阶段被拒绝。

## 调用说明

提供GE图模式调用；TensorFlow解析插件将 `tf.raw_ops.GRUBlockCellGrad` 映射为同名GE算子。

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | [test_geir_gru_block_cell_grad.cpp](examples/arch35/test_geir_gru_block_cell_grad.cpp) | 算子IR定义见[op_graph/gru_block_cell_grad_proto.h](op_graph/gru_block_cell_grad_proto.h)。 |
