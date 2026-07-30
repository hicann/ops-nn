# GruGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term>                            |     ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |     √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |     √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |     ×    |
| <term>Atlas 推理系列产品</term>                             |     ×     |
| <term>Atlas 训练系列产品</term>                              |     ×   |

## 功能说明

- 算子功能：门控循环单元（GRU）的**反向算子**，对应 [Gru](../gru/README.md) 前向。给定前向的输入与各门控中间结果，以及来自上层的输出梯度，计算输入、初始隐状态、权重和偏置的梯度。支持多层堆叠、双向、有/无偏置；定长 3D 输入。

- 计算思路（单层单向，BPTT，时间步从 `T-1` 到 `0`）：设 `hp = init_h (t=0)` 或 `output_h[t-1]`，门序为 `r, z, n`，则

  $$
  \text{grad\_h}_t = \text{dy}[t] + \text{dh}_{\text{next}}
  $$

  $$
  d n = \text{grad\_h}_t \odot (1 - z_t), \quad d z_{\text{raw}} = \text{grad\_h}_t \odot (h_p - n_t), \quad d h_{\text{prev\_from\_h}} = \text{grad\_h}_t \odot z_t
  $$

  $$
  d i_n = d n \odot (1 - n_t^2), \quad d r = (d i_n \odot h_n) \odot r_t \odot (1 - r_t), \quad d z = d z_{\text{raw}} \odot z_t \odot (1 - z_t)
  $$

  $$
  d g_i = [d r,\ d z,\ d i_n], \quad d g_h = [d r,\ d z,\ d i_n \odot r_t]
  $$

  聚合所有时间步后，通过 matmul 与 reduce 得到（权重为 kernel 布局 `w_input[I,3H]`、`w_hidden[H,3H]`）：

  $$
  d w_{input} = x^\top \cdot d g_i \quad [I, 3H], \qquad d w_{hidden} = h_p^\top \cdot d g_h \quad [H, 3H]
  $$

  $$
  d x = d g_i \cdot w_{input} \quad [T, B, I], \qquad d h_{prev} = d g_h \cdot w_{hidden} \quad [1, B, H]
  $$

  $$
  d b_{input} = \sum_{T,B} d g_i \quad [3H], \qquad d b_{hidden} = \sum_{T,B} d g_h \quad [3H]
  $$

  其中 $\odot$ 为逐元素乘法；`dh_next` 为下一时间步回传的隐状态梯度（`t=T-1` 时由输入 `dh` 提供）。

- 多层双向：由 aclnn `aclnnGRUBackward` 在 host 侧编排 —— 逐层（`L-1 → 0`）逐方向调用本算子，`dy` 沿特征维按方向切分，`dx` 双向求和并链式回传为下层 `dy`，`dh_prev`/`dw`/`db` 按层/方向路由。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>x</td>
      <td>输入</td>
      <td>前向输入序列，对应公式中的 $x_t$。定长 shape 为 (T, B, I)。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND、NCL</td>
    </tr>
    <tr>
      <td>w_input</td>
      <td>输入</td>
      <td>输入侧权重（kernel 布局），对应 $W_{ir}/W_{iz}/W_{in}$。形状为 $[I, 3H]$（首层）或 $[D*H, 3H]$（非首层）。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w_hidden</td>
      <td>输入</td>
      <td>隐状态侧权重（kernel 布局），对应 $W_{hr}/W_{hz}/W_{hn}$。形状为 $[H, 3H]$。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>init_h</td>
      <td>输入</td>
      <td>初始隐状态 $h_0$。形状为 $[1, B, H]$（单层单向）。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND、NCL</td>
    </tr>
    <tr>
      <td>output_h</td>
      <td>输入</td>
      <td>前向各步隐状态输出 $h_t$。形状为 (T, B, H)。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND、NCL</td>
    </tr>
    <tr>
      <td>reset_gate</td>
      <td>输入</td>
      <td>前向重置门激活值 $r_t$。形状为 (T, B, H)。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND、NCL</td>
    </tr>
    <tr>
      <td>update_gate</td>
      <td>输入</td>
      <td>前向更新门激活值 $z_t$。形状为 (T, B, H)。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND、NCL</td>
    </tr>
    <tr>
      <td>new_gate</td>
      <td>输入</td>
      <td>前向新门激活值 $n_t$。形状为 (T, B, H)。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND、NCL</td>
    </tr>
    <tr>
      <td>h_n</td>
      <td>输入</td>
      <td>前向隐状态侧新门预激活 $W_{hn}h_{t-1}+b_{hn}$。形状为 (T, B, H)。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND、NCL</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>各时间步输出梯度。形状为 (T, B, H)。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND、NCL</td>
    </tr>
    <tr>
      <td>dh</td>
      <td>输入</td>
      <td>末时刻隐状态梯度（来自下一时间步/上层）。形状为 [1, B, H]。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND、NCL</td>
    </tr>
    <tr>
      <td>batch_sizes</td>
      <td>可选输入</td>
      <td>不定长序列各时刻有效 batch 数。形状为 (T)。当前实现按定长处理。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>输入 x 的梯度。形状与 x 一致 (T, B, I)。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND、NCL</td>
    </tr>
    <tr>
      <td>dh_prev</td>
      <td>输出</td>
      <td>初始隐状态的梯度 $d h_0$。形状为 [1, B, H]。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND、NCL</td>
    </tr>
    <tr>
      <td>dw_input</td>
      <td>输出</td>
      <td>$d w_{input}$，形状与 w_input 一致。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dw_hidden</td>
      <td>输出</td>
      <td>$d w_{hidden}$，形状与 w_hidden 一致。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>db_input</td>
      <td>输出</td>
      <td>$d b_{input}$，形状为 $[3H]$。has_bias=false 时不计算。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>db_hidden</td>
      <td>输出</td>
      <td>$d b_{hidden}$，形状为 $[3H]$。has_bias=false 时不计算。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>direction</td>
      <td>属性</td>
      <td>GRU 方向，取值 "UNIDIRECTIONAL"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_bias</td>
      <td>属性</td>
      <td>是否计算偏置梯度。false 时 kernel 跳过 bias 梯度计算、不输出有效 db。默认 true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_training</td>
      <td>属性</td>
      <td>是否训练模式（反向恒为 true，与前向对齐）。默认 true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_first</td>
      <td>属性</td>
      <td>input/dy/dx 的 batch 是否在第一维。当前 aclnn 层拒绝 true。默认 false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入 x 为 3D 定长 (T, B, I)；`batch_first=true` 与真正的不定长（PackedSequence）暂不支持。
- 权重为 kernel 布局：`w_input` 为 $[I, 3H]$（首层）或 $[D*H, 3H]$（非首层），`w_hidden` 为 $[H, 3H]$；aclnn 入参的 `params` 按 PyTorch 布局 $[3H, *]$ 传入。
- 门控输入（reset_gate/update_gate/new_gate/h_n/output_h/dy）均要求 (T, B, H) 且 dtype 与 x 一致。
- `has_bias=false` 时不计算 bias 梯度（`db_input`/`db_hidden` 无有效输出）。
- x/dy/dh/output_h 及各门控支持 ND 与 NCL 两种格式；`batch_sizes`（1D）仅支持 ND。
