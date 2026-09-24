# Gru

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

- 算子功能：实现门控循环单元（Gated Recurrent Unit, GRU）计算，支持多层堆叠、双向、定长序列和不定长序列（PackedSequence）两种输入模式。训练模式下可输出各门控中间结果（r、z、n、n_h），用于反向传播。

- 输入输出shape：
  - 定长模式：输入x的shape为(T, B, I)或(B, T, I)，输出y的shape为(T, B, D*H)或(B, T, D*H)，最终隐状态output_h的shape为(L*D, B, H)。
  - 不定长模式：输入x的shape为(sum(batch_size), I)，输出y的shape为(sum(batch_size), D*H)，最终隐状态output_h的shape为(L*D, B, H)。目前暂不支持。

- 计算公式：

  $$
  r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr})
  $$

  $$
  z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz})
  $$

  $$
  n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{(t-1)} + b_{hn}))
  $$

  $$
  h_t = (1 - z_t) \odot n_t + z_t \odot h_{(t-1)}
  $$

  其中：
  - $\sigma$ 为 sigmoid 激活函数。
  - $\odot$ 表示逐元素乘法。
  - $r_t$、$z_t$、$n_t$ 分别为重置门、更新门和新门。
  - $h_t$ 为当前时刻的隐状态。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 表示输入的序列数据，对应公式中的 $x_t$。定长模式shape为(T, B, I)或(B, T, I)，不定长模式shape为(sum(batch_size), I)。 | FLOAT32、FLOAT16 | ND |
| wi | 输入 | 表示输入权重矩阵，对应公式中的 $W_{ir}$、$W_{iz}$、$W_{in}$。形状为 $[3H, I]$（首层）或 $[3H, D*H]$（非首层）。 | FLOAT32、FLOAT16 | ND |
| wh | 输入 | 表示隐状态权重矩阵，对应公式中的 $W_{hr}$、$W_{hz}$、$W_{hn}$。形状为 $[3H, H]$。 | FLOAT32、FLOAT16 | ND |
| bi | 可选输入 | 表示输入偏置，对应公式中的 $b_{ir}$、$b_{iz}$、$b_{in}$。形状为 $[3H]$。 | FLOAT32、FLOAT16 | ND |
| bh | 可选输入 | 表示隐状态偏置，对应公式中的 $b_{hr}$、$b_{hz}$、$b_{hn}$。形状为 $[3H]$。 | FLOAT32、FLOAT16 | ND |
| batch_sizes | 可选输入 | 表示不定长序列的batch大小数组，对应PackedSequence模式。形状为(T)。 | INT64 | ND |
| init_h | 可选输入 | 表示初始隐状态，对应公式中的 $h_0$。形状为(L*D, B, H)。 | FLOAT32、FLOAT16 | ND |
| y | 输出 | 表示所有时间步的输出，对应公式中的 $h_t$。定长模式shape为(T, B, D*H)或(B, T, D*H)，不定长模式shape为(sum(batch_size), D*H)。 | FLOAT32、FLOAT16 | ND |
| output_h | 输出 | 表示最后一层所有方向的最终隐状态，对应公式中最后时刻的 $h_t$。形状为(L*D, B, H)。 | FLOAT32、FLOAT16 | ND |
| r | 输出 | 表示重置门输出，对应公式中的 $r_t$。定长模式shape为(T, B, H)，不定长模式shape为(sum(batch_size), H)。 | FLOAT32、FLOAT16 | ND |
| z | 输出 | 表示更新门输出，对应公式中的 $z_t$。定长模式shape为(T, B, H)，不定长模式shape为(sum(batch_size), H)。 | FLOAT32、FLOAT16 | ND |
| n | 输出 | 表示新门输出，对应公式中的 $n_t$。定长模式shape为(T, B, H)，不定长模式shape为(sum(batch_size), H)。 | FLOAT32、FLOAT16 | ND |
| n_h | 输出 | 表示隐状态-新门输出 $W_{hn} h_{(t-1)} + b_{hn}$。定长模式shape为(T, B, H)，不定长模式shape为(sum(batch_size), H)。 | FLOAT32、FLOAT16 | ND |
| direction | 属性 | 表示GRU方向，取值"UNIDIRECTIONAL"表示单向。 | STRING | - |
| is_training | 属性 | 表示是否为训练模式，默认值为true。 | BOOL | - |

## 约束说明

- 输入x的维度只能为2维（不定长模式）或3维（定长模式）。
- 权重wi的首层形状为 $[3H, I]$，非首层形状为 $[3H, D*H]$。
- 权重wh的形状为 $[3H, H]$。
- 偏置bi和bh的形状为 $[3H]$。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_gru](examples/test_aclnn_gru.cpp) | 通过[aclnnGRU](docs/aclnnGRU.md)接口方式调用Gru算子。 |
