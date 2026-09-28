# ThnnFusedGruCellGrad

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :--- |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：完成 GRU（门控循环单元）单时间步的融合反向梯度计算。输入上游梯度 `grad_hy (B, H)` 与前向 `_thnn_fused_gru_cell` 保存的中间结果 `storage (B, 5H)`（行内按 `[r, z, n, hx, hn]` 五个 H 宽平面顺序存放），单 kernel 融合计算 5 路梯度输出——两路门预激活梯度、隐状态直连梯度、两路偏置梯度。对标 PyTorch `aten::_thnn_fused_gru_cell_backward`。
- 计算公式：

  记 $go = \mathrm{grad\_hy}$，从 `storage` 行内拆分五个平面 $r, z, n, hx, hn$（各 $H$ 宽），五条梯度链为：

  $$\mathrm{gin} = go \cdot (1 - z) \cdot (1 - n^2) \tag{tanh\_backward}$$

  $$\mathrm{gig} = go \cdot (hx - n) \cdot (1 - z) \cdot z \tag{sigmoid\_backward}$$

  $$\mathrm{grg} = \mathrm{gin} \cdot hn \cdot (1 - r) \cdot r \tag{sigmoid\_backward}$$

  $$\mathrm{ghn} = \mathrm{gin} \cdot r$$

  $$\mathrm{ghx} = go \cdot z$$

  输出拼接为 `grad_input_gates = [grg, gig, gin]`、`grad_hidden_gates = [grg, gig, ghn]`（均 $(B, 3H)$）、`grad_hx = ghx`（$(B, H)$）；两路 bias 为对应门梯度沿 batch 维归约（各 $(3H,)$）。`has_bias = false` 时两路 bias 输出为空张量 $(0,)$。

## 参数说明

|参数名|输入/输出/属性|描述|数据类型|数据格式|
|-----|-----------|----|---------|------|
|grad_hy|输入|上游梯度 ∂L/∂hy（前向输出 hy 的梯度），shape 为 (B, H)，对应公式中 go。|FLOAT、FLOAT16、BFLOAT16|ND|
|storage|输入|前向 _thnn_fused_gru_cell 保存的中间结果，行内按 [r, z, n, hx, hn] 五平面存放，shape 为 (B, 5H)。数据类型与 grad_hy 保持一致。|FLOAT、FLOAT16、BFLOAT16|ND|
|has_bias|属性|表示前向 input_bias 是否定义（bool，默认 false）。true 时输出两路 bias 梯度 (3H,)；false 时输出空张量 (0,)。|BOOL|-|
|grad_input_gates|输出|input 侧门预激活梯度，行内 [grg, gig, gin] 三段拼接，shape 为 (B, 3H)。|FLOAT、FLOAT16、BFLOAT16|ND|
|grad_hidden_gates|输出|hidden 侧门预激活梯度，行内 [grg, gig, ghn] 三段拼接，shape 为 (B, 3H)。|FLOAT、FLOAT16、BFLOAT16|ND|
|grad_hx|输出|隐状态直连梯度，shape 为 (B, H)。|FLOAT、FLOAT16、BFLOAT16|ND|
|grad_input_bias|输出|grad_input_gates 沿 batch 维归约结果，shape 为 (3H,)（has_bias=true）或 (0,)（has_bias=false）。|FLOAT、FLOAT16、BFLOAT16|ND|
|grad_hidden_bias|输出|grad_hidden_gates 沿 batch 维独立归约结果，shape 为 (3H,)（has_bias=true）或 (0,)（has_bias=false）。|FLOAT、FLOAT16、BFLOAT16|ND|

## 约束说明

- 数据类型：grad_hy 与 storage 数据类型须一致，仅支持 FLOAT、FLOAT16、BFLOAT16。所有输出数据类型与 grad_hy 保持一致。
- 数据格式：输入和输出仅支持 ND。
- 维度约束：grad_hy 与 storage 须为 2 维，且 storage 的 shape 须满足 (B, 5H)（第二维为 H 的 5 倍，B 维与 grad_hy 一致）。
- 非连续 Tensor：输入与输出均支持非连续 Tensor——输入经 Contiguous 归一为连续后计算；输出经 ViewCopy 按用户视图散射写回（空输出跳过）。
- 空张量：支持空张量输入。B=0 或 H=0 为合法退化——门梯度与 grad_hx 输出为空；has_bias=true 且 B=0 时两路 bias 输出为全零 (3H,)；has_bias=false 时两路 bias 输出为空张量 (0,)。
- 确定性：默认确定性实现，同输入多次运行结果位级一致。

## 调用说明

- <term>Ascend 950PR&950DT系列产品</term> ：

    | 调用方式 | 调用样例 | 说明 |
    |---------|---------|------|
    | aclnn API | [test_aclnn_thnn_fused_gru_cell_grad.cpp](examples/arch35/test_aclnn_thnn_fused_gru_cell_grad.cpp) | 通过[aclnnThnnFusedGruCellBackward](docs/aclnnThnnFusedGruCellBackward.md)接口调用ThnnFusedGruCellGrad算子 |
    | GE图模式 | - | 通过[算子IR](op_graph/thnn_fused_gru_cell_grad_proto.h)构图方式调用ThnnFusedGruCellGrad算子 |
