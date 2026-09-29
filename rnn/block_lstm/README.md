# BlockLstm

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

- 算子功能：实现统一的TF BlockLSTM（V1+V2）前向计算。在T个时间步上计算LSTM的细胞状态和隐状态，同时输出每个时间步的输入门、遗忘门、候选状态和输出门。
- 计算公式：

  计算门控激活值：

  $$
  \begin{aligned}
  gates &= x \cdot w_{ix} + h_{prev} \cdot w_{hh} + b \\
  i_{out} &= \sigma(gates_{i}) \\
  f_{out} &= \sigma(gates_{f} + forgetBias) \\
  g_{out} &= \tanh(gates_{g}) \\
  o_{out} &= \sigma(gates_{o})
  \end{aligned}
  $$

  更新细胞状态：

  $$
  \begin{aligned}
  ci_{out} &= g_{out} \\
  cs_{out} &= f_{out} \odot cs_{prev} + i_{out} \odot ci_{out}
  \end{aligned}
  $$

  若 cellClip > 0，则对细胞状态进行裁剪：

  $$
  cs_{out} = \min(\max(cs_{out}, -cellClip), cellClip)
  $$

  更新隐状态：

  $$
  \begin{aligned}
  co_{out} &= \tanh(cs_{out}) \\
  h_{out} &= o_{out} \odot co_{out}
  \end{aligned}
  $$

  若 usePeephole 为 true，则门控计算中引入窥孔连接：

  $$
  \begin{aligned}
  i_{out} &= \sigma(gates_{i} + wci \odot cs_{prev}) \\
  f_{out} &= \sigma(gates_{f} + forgetBias + wcf \odot cs_{prev}) \\
  o_{out} &= \sigma(gates_{o} + wco \odot cs_{out})
  \end{aligned}
  $$

  - gate_order="ifco" 对应 TF BlockLSTMV2；gate_order="icfo" 对应 TF BlockLSTM (V1)
  - $\sigma$ 为Sigmoid激活函数，$\odot$ 为逐元素乘积

## 参数说明

<table style="undefined;table-layout: fixed; width: 885px"><colgroup>
  <col style="width: 194px">
  <col style="width: 138px">
  <col style="width: 300px">
  <col style="width: 133px">
  <col style="width: 120px">
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
      <td>seq_len_max</td>
      <td>输入</td>
      <td>实际序列长度最大值，用于对超出该长度的时间步进行掩码处理。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>输入序列，即每个时间步的输入数据。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cs_prev</td>
      <td>输入</td>
      <td>初始细胞状态。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h_prev</td>
      <td>输入</td>
      <td>初始隐状态。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w</td>
      <td>输入</td>
      <td>融合权重矩阵，包含输入权重和隐藏权重。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wci</td>
      <td>输入</td>
      <td>输入门窥孔权重。usePeephole为false时传入全零向量。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wcf</td>
      <td>输入</td>
      <td>遗忘门窥孔权重。usePeephole为false时传入全零向量。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wco</td>
      <td>输入</td>
      <td>输出门窥孔权重。usePeephole为false时传入全零向量。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>输入</td>
      <td>偏置。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>forget_bias</td>
      <td>属性</td>
      <td>遗忘门偏置加成。gate_order="icfo"（V1）默认1.0，gate_order="ifco"（V2）默认0.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cell_clip</td>
      <td>属性</td>
      <td>细胞状态裁剪阈值。为0时不裁剪。gate_order="icfo"（V1）默认3.0，gate_order="ifco"（V2）默认0.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_peephole</td>
      <td>属性</td>
      <td>是否使用窥孔连接。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gate_order</td>
      <td>属性</td>
      <td>门控顺序。"ifco"对应TF BlockLSTMV2，"icfo"对应TF BlockLSTM (V1)。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>i</td>
      <td>输出</td>
      <td>每个时间步的输入门激活值。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cs</td>
      <td>输出</td>
      <td>每个时间步的细胞状态。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>f</td>
      <td>输出</td>
      <td>每个时间步的遗忘门激活值。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>o</td>
      <td>输出</td>
      <td>每个时间步的输出门激活值。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ci</td>
      <td>输出</td>
      <td>每个时间步的候选状态。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>co</td>
      <td>输出</td>
      <td>每个时间步的tanh(cs)值。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h</td>
      <td>输出</td>
      <td>每个时间步的隐状态。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 仅支持Ascend 950PR&950DT系列产品。
- 所有输入、输出参数的数据类型需保持一致，支持FLOAT和FLOAT16。
- 输入维度约束：x仅支持3维，shape为(T, B, I)；cs_prev/h_prev仅支持2维，shape为(B, H)；w为2维(I+H, 4H)；wci/wcf/wco为1维(H)；b为1维(4H)。维度或shape不符时算子拒绝执行（与TF BlockLSTM/BlockLSTMV2一致：x必须为3维、cs_prev必须为2维）。
- seq_len_max须满足 0 <= seq_len_max <= T，超出T的时间步输出置零。
- wci/wcf/wco在use_peephole=false时须传入全零向量。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| GEIR调用 | [test_geir_block_lstm](examples/arch35/test_geir_block_lstm.cpp) | 通过GEIR图模式调用BlockLstm算子（静态shape，含CPU golden数值校验）。 |
| GEIR动态 | [test_geir_block_lstm_dynamic](examples/arch35/test_geir_block_lstm_dynamic.cpp) | 同图同Session连续运行未知维(-1)与未知Rank(-2)场景，校验全部输出shape/dtype/数值。 |
