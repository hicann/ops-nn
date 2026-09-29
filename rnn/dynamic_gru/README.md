# DynamicGRU

DynamicGRU 是单层、单向 GRU（门控循环单元）前向整段循环算子：一次调用完成全部时间步的「双门 GEMM + 候选门 GEMM + 门控逐元素更新」，输出逐步隐状态 `y`、末步隐状态 `output_h` 与 `r`/`i`/`n` 三路门激活；并吸收 RnnGenMaskV2 的变长序列掩码能力（`seq_length` 传入即激活，掩码生成与冻结式施加在算子内部完成）。

> 本算子不提供 aclnn 单算子接口（算子工程与安装包均不含 `op_api/`），调用通路为 GEIR / GE 图模式（见 [调用说明](#调用说明) 与 [调用示例](examples/test_geir_dynamic_gru.cpp)）；PyTorch 侧可经 TorchAir GE raw op 在图模式下调用。

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：完成单层、单向 GRU 前向整段循环计算（CANN DynamicGRU V1 语义：reset 前置候选门、`w` 双门联合权重列序 [r, i]、`cw` 候选权重独立拆分），并内化 seqlength 融合规则的「原始长度 → 掩码 → 冻结式施加」语义。
- 计算公式（对每个时间步 t；σ 为 sigmoid，⊙ 为逐元素乘，[·;·] 为列拼接）：

  $$
  gates_t = [\,x_t\,;\,h_{t-1}\,]\cdot w + b_{[0:2H]}
  $$

  $$
  r_t = \sigma(gates_t[:,\,0:H]),\qquad i_t = \sigma(gates_t[:,\,H:2H])
  $$

  $$
  n_t = \tanh([\,x_t\,;\,r_t \odot h_{t-1}\,]\cdot cw + cb_{[0:H]})
  $$

  $$
  h_t = (h_{t-1} - n_t)\odot i_t + n_t
  $$

  `seq_length` 传入时按 $mask_t[b] = (t < seq\_length[b])$ 生成 0/1 掩码并冻结式施加：$h_t = h_{t-1} + (h_t - h_{t-1})\odot mask_t$（padding 步隐状态冻结在边界值，`r`/`i`/`n` 不受掩码影响）；`y_t = h_t`，`output_h` 为末步（掩码激活时为有效长度边界处）隐状态。
- 数值口径：两次 GEMM 的乘加在 fp32 累加，sigmoid/tanh 在 fp32 计算后按输出 dtype 回转。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|----------------|------|----------|----------|
| x | 输入 | 时间步输入序列 (num_step, batch_size, input_size)，time_major | FLOAT16 | ND |
| w | 输入 | r/i 双门联合权重 [input_size+hidden_size, 2*hidden_size]，输出列序 [r, i] | FLOAT16 | ND |
| b | 输入 | 双门偏置（1D，长度不小于 2*hidden_size 且 16 对齐，有效值为前 2*hidden_size 个）；须与 cb 同 dtype | FLOAT16、FLOAT | ND |
| cw | 输入 | 候选门权重 [input_size+hidden_size, hidden_size] | FLOAT16 | ND |
| cb | 输入 | 候选门偏置（1D，长度不小于 hidden_size 且 16 对齐，有效值为前 hidden_size 个）；须与 b 同 dtype | FLOAT16、FLOAT | ND |
| seq_length | 可选输入 | 逐 batch 真实序列长度 [batch_size]，值域 0 ≤ s[b] ≤ num_step；传入即激活掩码能力，缺省为定长行为 | INT32 | ND |
| init_h | 可选输入 | 初始隐状态 (hidden_size, batch_size)；缺省时首步 h 置零；须与 b/cb 同 dtype | FLOAT16、FLOAT | ND |
| y | 输出 | 逐步隐状态 (num_step, hidden_size, batch_size)，y_t = h_t（掩码激活时为冻结后的 h_t）；dtype 跟随 b/cb | FLOAT16、FLOAT | ND |
| output_h | 输出 | 末步隐状态，与 y 同 shape 逐步写回（掩码激活时为有效长度边界处隐状态）；dtype 跟随 b/cb | FLOAT16、FLOAT | ND |
| r | 输出 | 重置门激活 σ(·)，值域 (0, 1)，不受掩码影响；dtype 跟随 b/cb | FLOAT16、FLOAT | ND |
| i | 输出 | 更新门激活 σ(·)，值域 (0, 1)，不受掩码影响；dtype 跟随 b/cb | FLOAT16、FLOAT | ND |
| n | 输出 | 候选门激活 tanh(·)，值域 (-1, 1)，不受掩码影响；dtype 跟随 b/cb | FLOAT16、FLOAT | ND |
| direction | 属性 | 方向，默认 "UNIDIRECTIONAL"，仅支持该值（双向需拆两个方向各调一次） | String | - |
| cell_depth | 属性 | cell 深度，默认 1，仅支持 1（占位属性） | Int | - |
| keep_prob | 属性 | dropout 保留率，默认 1.0，仅支持 1.0（占位属性，无 dropout） | Float | - |
| cell_clip | 属性 | cell 裁剪，默认 -1.0，仅支持 -1.0（占位属性，不裁剪） | Float | - |
| num_proj | 属性 | 投影维度，默认 0，仅支持 0（占位属性，无投影） | Int | - |
| time_major | 属性 | 时间主序布局，默认 true，仅支持 true | Bool | - |
| activation | 属性 | 激活函数，默认 "tanh"，仅支持 "tanh" | String | - |
| is_training | 属性 | 训练态标记，默认 true，true/false 均可（占位属性，前向语义不区分） | Bool | - |

受支持 dtype 组合共 2 组：`x`/`w`/`cw` 恒 FLOAT16、`seq_length` 恒 INT32；差异仅在 `b`/`cb`（及跟随其 dtype 的 `init_h` 与全部输出）取 FLOAT16 或 FLOAT32。

## 约束说明

- shape 硬约束：`w.shape[0] == x.shape[2] + cw.shape[1]`；`w.shape[1] == 2 * cw.shape[1]`；`cw.shape == (input_size+hidden_size, hidden_size)`；`x.shape[1] == seq_length.shape[0] == init_h.shape[1]`；`init_h.shape[0] == cw.shape[1]`；5 个输出 shape 全等为 (L, H, N)。输入间无 numpy 式广播。
- dtype 约束：仅支持上述 2 个组合；`b` 与 `cb` 必须同 dtype，`init_h` 与全部输出必须与 `b`/`cb` 同 dtype。
- 数据格式：全部输入与输出仅支持 ND。
- 对齐约束：`b` 长度不小于 2H 且 16 对齐；`cb` 长度不小于 H 且 16 对齐。
- 空 Tensor：K = 0（I = 0 且 H = 0）时输出为空 Tensor；L = 0 或 N = 0 时输出自然为空；非空路径要求 input_size I > 0。
- 特殊值：入口不拒绝 NaN/Inf——NaN 沿时间传播，±Inf 门激活饱和、输出有界。
- 确定性：时间步间严格串行递推、无原子写，同输入同设备多次执行结果 bitwise 一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| aclnn API | - | 本算子不提供 aclnn 单算子接口（无 `op_api/`） |
| GE图模式 | [test_geir_dynamic_gru.cpp](examples/test_geir_dynamic_gru.cpp) | 通过[算子IR定义](op_graph/dynamic_gru_proto.h)构图调用（主通路）：安装本算子包后，包含算子 proto 头构建 GE 图并经 `ge::Session` 执行，参见调用示例 |
| PyTorch API | - | TorchAir GE 图模式：经 `torchair.ge.custom_op("DynamicGRU", ...)`（torchair 按注册 IR 自动生成的同名 raw op，签名与 IR 一致）在 `torch.compile` 图模式下发；无 eager 通路 |

## 参考资源

- [算子 IR 定义（op_graph/dynamic_gru_proto.h）](op_graph/dynamic_gru_proto.h)
- [GE 图模式调用示例（examples/test_geir_dynamic_gru.cpp）](examples/test_geir_dynamic_gru.cpp)
