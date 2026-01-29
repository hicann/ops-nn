# BidirectionLSTM

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>昇腾910_95 AI处理器</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas 200/300/500 推理产品</term>       |     ×    |

## 功能说明

- 算子功能：LSTM（Long Short-Term Memory，长短时记忆）网络是一种特殊的循环神经网络（RNN）模型。进行LSTM网络计算，接收输入序列和初始状态，返回输出序列和最终状态。
- 计算公式：
  
  $$
  f_t =sigm(W_f[h_{t-1}, x_t] + b_f)\\
  i_t =sigm(W_i[h_{t-1}, x_t] + b_i)\\
  o_t =sigm(W_o[h_{t-1}, x_t] + b_o)\\
  \tilde{c}_t =tanh(W_c[h_{t-1}, x_t] + b_c)\\
  c_t =f_t ⊙ c_{t-1} + i_t ⊙ \tilde{c}_t\\
  c_{o}^{t} =tanh(c_t)\\
  h_t =o_t ⊙ c_{o}^{t}\\
  $$

  - $x_t ∈ R^{d}$：LSTM单元的输入向量。
  - $f_t ∈ (0, 1)^{h}$：遗忘门激活向量。
  - $i_t ∈ (0, 1)^{h}$：输入门、更新门激活向量。
  - $o_t ∈ (0, 1)^{h}$：输出门激活向量。
  - $h_i ∈ (-1, 1)^{h}$：隐藏状态向量，也称为LSTM单元的输出向量。
  - $\tilde{c}_t ∈ (-1, 1)^{h}$：cell输入激活向量。
  - $c_t ∈ R^{h}$：cell状态向量。
  - $W ∈ R^{h×d}，(U ∈ R^{h×h})∩(b ∈ R^{h})$：训练中需要学习的权重矩阵和偏置向量参数。

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
      <td>x</td>
      <td>输入</td>
      <td>LSTM单元的输入向量，公式中的x。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>initH</td>
      <td>输入</td>
      <td>初始化hidden状态，公式中的h。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>initC</td>
      <td>输入</td>
      <td>初始化cell状态，公式中的c。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>wIh</td>
      <td>输入</td>
      <td>input-hidden权重，公式中的W。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>wHh</td>
      <td>输入</td>
      <td>hidden-hidden权重，公式中的W。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>bIhOptional</td>
      <td>输入</td>
      <td>input-hidden偏移，公式中的b。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>bHhOptional</td>
      <td>输入</td>
      <td>hidden-hidden偏移，公式中的b。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>wIhReverseOptional</td>
      <td>输入</td>
      <td>逆向input-hidden权重，公式中的W。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>wHhReverseOptional</td>
      <td>输入</td>
      <td>逆向hidden-hidden权重，公式中的W。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>bIhReverseOptional</td>
      <td>输入</td>
      <td>逆向input-hidden偏移，公式中的b。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>bHhReverseOptional</td>
      <td>输入</td>
      <td>逆向hidden-hidden偏移，公式中的b。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>numLayers</td>
      <td>属性</td>
      <td>表示LSTM层数。当前只支持1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
      <tr>
      <td>isbias</td>
      <td>属性</td>
      <td>表示是否有bias。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
      <tr>
      <td>batchFirst</td>
      <td>属性</td>
      <td>表示batch是否是第一维。当前只支持false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
      <tr>
      <td>bidirection</td>
      <td>属性</td>
      <td>表示是否是双向。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
       <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>LSTM单元的输出向量。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>outputHOut</td>
      <td>输出</td>
      <td>最终hidden状态，公式中的h。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>outputCOut</td>
      <td>输出</td>
      <td>最终cell状态，公式中的c。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_bidirection_lstm](./examples/test_aclnn_bidirection_lstm.cpp) | 通过[aclnnBidirectionLSTM](./docs/aclnnBidirectionLSTM.md)接口方式调用BidirectionLSTM算子。    |
| 图模式调用 | -  | 通过[算子IR](./op_graph/bidirection_lstm_proto.h)构图方式调用BidirectionLSTM算子。 |
