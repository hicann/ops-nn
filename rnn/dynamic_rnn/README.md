# DynamicRnn

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
 

## 功能说明

- 算子功能：基础循环神经网络 (Recurrent Neural Network) 算子，用于处理序列数据。它通过隐藏状态传递时序信息，适合处理具有时间/顺序依赖性的数据。
- 算子公式：
以LSTM格式为例给定输入 $x_t$、前一时刻隐藏状态 $h_{t-1}$ 和细胞状态 $c_{t-1}$，DynamicRNN的计算过程如下：

1. **遗忘门** (Forget Gate):
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

2. **输入门** (Input Gate):
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

3. **细胞状态更新**:
   $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

4. **输出门** (Output Gate):
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(c_t)$$

其中：
- $\sigma$ 是 sigmoid 函数
- $\odot$ 表示逐元素乘法 (Hadamard product)
- $W_*$ 是可学习的权重矩阵
- $b_*$ 是可学习的偏置项

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 20px">
  <col style="width: 30px">
  <col style="width: 150px">
  <col style="width: 50px">
  <col style="width: 50px">
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
      <td>输入数据（T， Batch, input_size）。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w</td>
      <td>输入</td>
      <td>输入权重 （input_size + hidden_size, 4 * hidden_size）。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>输入</td>
      <td>输入偏置 （4 * hidden_size）。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seq_length</td>
      <td>输入</td>
      <td>每个batch真实的长度，维度为（T， Batch, input_size）。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>init_h</td>
      <td>输入</td>
      <td>初始时刻的hidden state, 维度为(1, batch_size, hidden_size)。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>init_c</td>
      <td>输入</td>
      <td>初始时刻的cell state, 维度为(1, batch_size, hidden_size)。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输入</td>
      <td>输入的掩码矩阵。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出结果，维度为（T， batch_size, hidden_size）。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_h</td>
      <td>输出</td>
      <td>输出结果，维度为（T， batch_size, hidden_size）。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_c</td>
      <td>输出</td>
      <td>输出结果，维度为（T， batch_size, hidden_size）。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>i</td>
      <td>输出</td>
      <td>输出结果，维度为（T， batch_size, hidden_size）。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>j</td>
      <td>输出</td>
      <td>输出结果，维度为（T， batch_size, hidden_size）。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>f</td>
      <td>输出</td>
      <td>输出结果，维度为（T， batch_size, hidden_size）。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>o</td>
      <td>输出</td>
      <td>输出结果，维度为（T， batch_size, hidden_size）。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>tanhc</td>
      <td>输出</td>
      <td>输出结果，维度为（T， batch_size, hidden_size）。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cell_type</td>
      <td>属性</td>
      <td>默认为"LSTM", 当前实现"LSTM"、"GRU"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>direction</td>
      <td>属性</td>
      <td>默认为"UNIDIRECTIONAL"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cell_depth</td>
      <td>属性</td>
      <td>multi_rnn的级数，默认为1, 且当前只支持1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_peephole</td>
      <td>属性</td>
      <td>是否使用peephole, 默认false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keep_prob</td>
      <td>属性</td>
      <td>默认1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cell_clip</td>
      <td>属性</td>
      <td>默认值为-1.0(没有剪切)。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_proj</td>
      <td>属性</td>
      <td>用projection的降维后的维度，默认为0表示不降维。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>time_major</td>
      <td>属性</td>
      <td>输入数据的排列方式，默认为True。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activation</td>
      <td>属性</td>
      <td>激活函数"tanh"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>forget_bias</td>
      <td>属性</td>
      <td>默认1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gate_order</td>
      <td>属性</td>
      <td>默认"ijfo"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_training</td>
      <td>属性</td>
      <td>默认true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 |  - | 通过[算子IR](op_graph/dynamic_rnn_proto.h)构图方式调用DynamicRnn算子。         |

<!-- [test_geir_dynamic_rnn](examples/test_geir_dynamic_rnn.cpp) -->