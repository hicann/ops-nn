# DynamicAUGRUGrad

## 产品支持情况

| 产品                                            | 是否支持 |
| :---------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>         |    √     |
| <term>Atlas A3系列产品</term>                   |    ×     |
| <term>Atlas A2系列产品</term>                   |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>          |    ×     |
| <term>Atlas推理系列产品</term>                  |    ×     |
| <term>Atlas训练系列产品</term>                  |    ×     |

## 功能说明

- 算子功能：带注意力更新门的GRU（**AUGRU**）**反向算子**。给定前向的输入、权重、注意力与各门控中间结果以及上层梯度，按 BPTT（时间步从 `T-1` 到 `0`）计算输入、初始隐状态、权重与偏置的梯度，并额外输出注意力梯度 `dw_att`。传入 `seq_length` 时在 kernel 内在线生成变长掩码（`seq_mask[t,b] = (t < seq_length[b]) ? 1 : 0`），不传时等价全 1 掩码。

- 计算公式（单层单向，gate_order 为 zrh；rzh 时 z/r 槽位互换），设 $u_t = z_t \cdot (1 - \text{att}_t)$，按时间步倒序：

  $$
  \text{grad\_h}_t = \text{mask}_t \odot \text{dh}_{\text{prev}} + \text{dy}[t]
  $$

  $$
  \text{dnt} = \text{grad\_h}_t \odot (1 - u_t) \odot (1 - n_t^2), \qquad \text{dr} = \text{dnt} \odot r_t \odot (1 - r_t) \odot \tilde{n}_t
  $$

  $$
  \text{dz} = \text{grad\_h}_t \odot (h_{t-1} - n_t) \odot (1 - \text{att}_t) \odot z_t \odot (1 - z_t), \qquad \text{dw\_att}_t = -\sum_h (\text{grad\_h}_t \odot (h_{t-1} - n_t) \odot z_t)
  $$

  $$
  \text{dh}_{\text{prev}} = [\text{dz}, \text{dr}, \text{dnt} \odot r_t] \cdot W_{hh}^T + \text{grad\_h}_t \odot u_t
  $$

  循环结束后聚合（权重为 kernel 布局 `w_input[I,3H]`、`w_hidden[H,3H]`）：

  $$
  d w_{input} = x^T \cdot [\text{dz}, \text{dr}, \text{dnt}], \quad d w_{hidden} = h_{prev}^T \cdot [\text{dz}, \text{dr}, \text{dnt} \odot r_t], \quad d x = [\text{dz}, \text{dr}, \text{dnt}] \cdot W_{in}^T
  $$

  $$
  d b_{input} = \sum_{T,B} [\text{dz}, \text{dr}, \text{dnt}], \qquad d b_{hidden} = \sum_{T,B} [\text{dz}, \text{dr}, \text{dnt} \odot r_t]
  $$

## 参数说明

下表中，`T`表示序列长度，`B`表示批大小，`I`表示输入特征维度，`H`表示隐状态维度。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ------ | -------------- | ---- | -------- | -------- |
| x | 输入 | 表示前向输入序列，shape为 `[T, B, I]`。 | FLOAT32、FLOAT16 | ND |
| weight_input | 输入 | 表示输入侧权重，shape为 `[I, 3H]`。 | FLOAT32、FLOAT16 | ND |
| weight_hidden | 输入 | 表示隐状态侧权重，shape为 `[H, 3H]` 或 `[1, H, 3H]`。两种shape使用相同的连续 `H * 3H` 布局。 | FLOAT32、FLOAT16 | ND |
| weight_att | 输入 | 表示已沿隐状态维度广播的注意力得分，shape为 `[T, B, H]`。 | FLOAT32、FLOAT16 | ND |
| y | 输入 | 表示前向输出，shape为 `[T, B, H]`。当前实现将其作为占位输入，不参与数值计算。 | FLOAT32、FLOAT16 | ND |
| init_h | 输入 | 表示初始隐状态，shape为 `[B, H]`。 | FLOAT32、FLOAT16 | ND |
| h | 输入 | 表示前向各时间步的隐状态输出，shape为 `[T, B, H]`，其中 `h[t]` 为第 `t` 个时间步的输出。 | FLOAT32、FLOAT16 | ND |
| dy | 输入 | 表示各时间步输出的梯度，shape为 `[T, B, H]`。 | FLOAT32、FLOAT16 | ND |
| dh | 输入 | 表示末时间步隐状态的梯度，shape为 `[B, H]`。 | FLOAT32、FLOAT16 | ND |
| update | 输入 | 表示前向更新门激活值 z<sub>t</sub>，shape为 `[T, B, H]`。 | FLOAT32、FLOAT16 | ND |
| update_att | 输入 | 表示注意力作用后的更新门 u<sub>t</sub> = z<sub>t</sub> · (1 - att<sub>t</sub>)，shape为 `[T, B, H]`。 | FLOAT32、FLOAT16 | ND |
| reset | 输入 | 表示前向重置门激活值 r<sub>t</sub>，shape为 `[T, B, H]`。 | FLOAT32、FLOAT16 | ND |
| new | 输入 | 表示前向新门激活值 n<sub>t</sub>，shape为 `[T, B, H]`。 | FLOAT32、FLOAT16 | ND |
| hidden_new | 输入 | 表示新门隐状态侧预激活值，shape为 `[T, B, H]`。 | FLOAT32、FLOAT16 | ND |
| seq_length | 可选输入 | 表示各batch的实际序列长度，shape为 `[B]`。传入该参数时，算子内部生成掩码；不传时等价于全1掩码。 | INT32 | ND |
| mask | 可选输入 | dropout预留占位，当前实现不参与数值计算。 | UINT8 | ND |
| dw_input | 输出 | 表示输入侧权重的梯度，shape为 `[I, 3H]`。 | FLOAT32、FLOAT16 | ND |
| dw_hidden | 输出 | 表示隐状态侧权重的梯度，shape为 `[H, 3H]`。 | FLOAT32、FLOAT16 | ND |
| db_input | 输出 | 表示输入侧偏置的梯度，shape为 `[3H]`。 | FLOAT32、FLOAT16 | ND |
| db_hidden | 输出 | 表示隐状态侧偏置的梯度，shape为 `[3H]`。 | FLOAT32、FLOAT16 | ND |
| dx | 输出 | 表示前向输入序列的梯度，shape为 `[T, B, I]`。 | FLOAT32、FLOAT16 | ND |
| dh_prev | 输出 | 表示初始隐状态的梯度，shape为 `[B, H]`。 | FLOAT32、FLOAT16 | ND |
| dw_att | 输出 | 表示注意力的梯度，shape为 `[T, B]`。 | FLOAT32、FLOAT16 | ND |
| direction | 可选属性 | 表示循环方向。默认值为 `"UNIDIRECTIONAL"`，当前仅支持默认值。 | STRING | - |
| cell_depth | 可选属性 | 表示循环层数。默认值为 `1`，当前仅支持默认值。 | INT | - |
| keep_prob | 可选属性 | 表示dropout保留概率。默认值为 `-1.0`，当前支持 `1.0` 和 `-1.0`，dropout不生效。 | FLOAT | - |
| cell_clip | 可选属性 | 表示cell裁剪阈值。默认值为 `-1.0`，当前仅支持默认值，cell裁剪不生效。 | FLOAT | - |
| num_proj | 可选属性 | 表示投影维度。默认值为 `0`，当前仅支持默认值，即不启用投影。 | INT | - |
| time_major | 可选属性 | 表示输入是否采用时间维在前的布局。默认值为 `true`，当前仅支持默认值。 | BOOL | - |
| gate_order | 可选属性 | 表示权重和梯度中门的排列顺序。默认值为 `"zrh"`，支持 `"zrh"` 和 `"rzh"`。 | STRING | - |
| reset_after | 可选属性 | 表示是否在矩阵乘后应用重置门。默认值为 `true`，当前仅支持默认值。 | BOOL | - |

## 约束说明

- x 为 3D 定长 [T, B, I]；I、H 为正数；H、I 任意（非 16 对齐时按 padded 布局处理，pad 区恒 0）。
- 支持 FLOAT32 与 FLOAT16；两种 dtype 复用同一条 fp32 cube 通路（fp16 仅在输入/输出边界做 Cast，精度以 fp32 累加为准）。
- 除 seq_length（INT32）、mask（UINT8）外全部浮点输入/输出与 x 的 dtype 一致。
- `seq_length` 元素满足 0 ≤ seq_length[b] ≤ T；越界值不额外校验，kernel 按 `t < seq_length[b]` 比较天然钳位（大于 T 等价 T、小于 0 等价 0）。
- 双向、多层（cell_depth>1）、投影（num_proj>0）暂不支持。
- `keep_prob` 仅接受 1.0/-1.0（dropout 不生效），`cell_clip` 仅接受 -1.0（clip 不生效），其余取值报错。

## 调用说明

算子以二进制方式发布（`op_host/config/ascend950/dynamic_augru_grad_binary.json`），通过 GE Graph 模式调用：

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| GE图模式 | [test_geir_dynamic_augru_grad](examples/arch35/test_geir_dynamic_augru_grad.cpp) | 通过[算子IR](op_graph/dynamic_augru_grad_proto.h)构图并调用DynamicAUGRUGrad算子。 |
| GE图模式 | [test_geir_dynamic_augru_grad_dynamic](examples/arch35/test_geir_dynamic_augru_grad_dynamic.cpp) | 通过[算子IR](op_graph/dynamic_augru_grad_proto.h)在同一张图中以动态Shape或动态Rank调用DynamicAUGRUGrad算子。 |
