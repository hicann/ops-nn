# DynamicAUGRU

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：DynamicAUGRU实现由注意力分数调节更新门的门控循环单元（Attention-based Update Gate GRU），用于处理用户行为等序列数据。算子按时间步更新隐藏状态，返回完整的隐藏状态序列及门控中间结果。

- 输入输出shape：记`T`为时间步数，`B`为批大小，`I`为输入特征维度，`H`为隐藏状态维度。输入`x`的shape为`[T, B, I]`，七个输出的shape均为`[T, B, H]`。

- 计算公式：时间下标从0开始，$h_{-1}$为`init_h[0]`；未提供初始状态时取零。每个时间步的输入投影和隐藏状态投影为：

  $$
  G^x_t = x_t W_x + b_x, \qquad G^h_t = h_{t-1} W_h + b_h
  $$

  其中，$W_x$、$W_h$分别对应`weight_input`、`weight_hidden`，$b_x$、$b_h$分别对应`bias_input`、`bias_hidden`，未提供的偏置按零处理。将投影的最后一维分为更新门、重置门和候选状态三段，下标分别记为$z$、$r$、$n$。`gate_order="zrh"`时存储顺序为$z,r,n$，`gate_order="rzh"`时为$r,z,n$。

  $$
  \begin{aligned}
  z_t &= \sigma(G^x_{t,z} + G^h_{t,z}) \\
  r_t &= \sigma(G^x_{t,r} + G^h_{t,r}) \\
  n_t &= \tanh(G^x_{t,n} + r_t \odot G^h_{t,n}) \\
  \widehat{z}_t &= (1-a_t) \odot z_t \\
  \widetilde{h}_t &= n_t + \widehat{z}_t \odot (h_{t-1}-n_t)
  \end{aligned}
  $$

  $\sigma$为Sigmoid函数，$\odot$为逐元素乘法，$a_t$为`weight_att[t]`，沿隐藏维度广播。

## 参数说明

所有张量使用ND格式。表中的FLOAT表示32位浮点数。

<table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 470px">
  <col style="width: 170px">
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
      <td>输入序列特征，对应公式中的x<sub>t</sub>，shape为[T, B, I]。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_input</td>
      <td>输入</td>
      <td>输入投影权重，对应公式中的W<sub>x</sub>，shape为[I, 3H]。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_hidden</td>
      <td>输入</td>
      <td>隐藏状态投影权重，对应公式中的W<sub>h</sub>，shape为[H, 3H]。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_att</td>
      <td>输入</td>
      <td>注意力分数，对应公式中的a<sub>t</sub>，shape为[T, B]。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias_input</td>
      <td>可选输入</td>
      <td>输入投影偏置，对应公式中的b<sub>x</sub>，shape为[3H]，缺省时按零处理。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias_hidden</td>
      <td>可选输入</td>
      <td>隐藏状态投影偏置，对应公式中的b<sub>h</sub>，shape为[3H]，缺省时按零处理。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seq_length</td>
      <td>可选输入</td>
      <td>序列控制输入。INT32类型时shape为[B]；FLOAT16类型时作为逐元素掩码，shape为[T, B, H]；缺省时不进行序列控制。</td>
      <td>INT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>init_h</td>
      <td>可选输入</td>
      <td>初始隐藏状态，shape为[1, B, H]，缺省时按零处理。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>应用序列控制后的隐藏状态序列，shape为[T, B, H]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_h</td>
      <td>输出</td>
      <td>与y相同的完整隐藏状态序列，shape为[T, B, H]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>update</td>
      <td>输出</td>
      <td>更新门z<sub>t</sub>，shape为[T, B, H]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>update_att</td>
      <td>输出</td>
      <td>注意力调节后的更新门z&#770;<sub>t</sub>，shape为[T, B, H]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reset</td>
      <td>输出</td>
      <td>重置门r<sub>t</sub>，shape为[T, B, H]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>new</td>
      <td>输出</td>
      <td>候选隐藏状态n<sub>t</sub>，shape为[T, B, H]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>hidden_new</td>
      <td>输出</td>
      <td>候选状态对应的隐藏投影G<sup>h</sup><sub>t,n</sub>，包含隐藏侧偏置，尚未乘以重置门，shape为[T, B, H]。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>direction</td>
      <td>可选属性</td>
      <td>计算方向，默认值为"UNIDIRECTIONAL"，仅支持该值。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cell_depth</td>
      <td>可选属性</td>
      <td>循环单元层数，默认值为1，仅支持该值。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keep_prob</td>
      <td>可选属性</td>
      <td>保留概率，默认值为1.0，仅支持该值，不执行Dropout。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cell_clip</td>
      <td>可选属性</td>
      <td>状态裁剪参数，默认值为-1.0，仅支持该值，不执行裁剪。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_proj</td>
      <td>可选属性</td>
      <td>投影降维参数，默认值为0，仅支持该值，不执行投影降维。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>time_major</td>
      <td>可选属性</td>
      <td>是否按时间维优先排列，默认值为true，仅支持该值。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activation</td>
      <td>可选属性</td>
      <td>候选状态激活函数，默认值为"tanh"，仅支持该值。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gate_order</td>
      <td>可选属性</td>
      <td>权重、偏置的门排列顺序，默认值为"zrh"，支持"zrh"和"rzh"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reset_after</td>
      <td>可选属性</td>
      <td>是否在隐藏投影及偏置相加后应用重置门，默认值为true，仅支持该值。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_training</td>
      <td>可选属性</td>
      <td>训练模式标记，默认值为true，支持true和false；两种取值均返回七个输出。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 仅支持单层、单向、时间维优先的前向计算。
- 运行时`T`、`B`、`I`、`H`均须大于0，不支持空张量。各输入的关联维度须满足参数说明中的shape关系。
- 调用时输入按连续ND布局组织；本目录未提供非连续输入的专项调用示例或验证结果。
- `bias_input`、`bias_hidden`和`init_h`中已提供的张量须使用相同数据类型，七个输出与其保持一致。三者均未提供时，输出类型为FLOAT16。
- 未提供`seq_length`时，`y`和`output_h`保存当前时间步计算得到的隐藏状态。
- `seq_length`为INT32张量`[B]`时，对于第`b`个样本的第`t`个时间步：
  - 当`t < seq_length[b]`时，`y[t, b, :]`和`output_h[t, b, :]`保存当前时间步计算得到的隐藏状态。
  - 当`t >= seq_length[b]`时，沿用前一时间步的隐藏状态；`t`为0时使用`init_h[0, b, :]`，未提供`init_h`时使用全零状态。
  - `seq_length[b]`小于等于0时，全程保持初始状态；大于等于`T`时，全部时间步参与计算。
- `seq_length`为FLOAT16张量`[T, B, H]`时，`seq_length[t, b, h]`作为对应隐藏状态元素的混合系数。取值为0时沿用前一时间步的隐藏状态，取值为1时使用当前时间步计算得到的隐藏状态，其他数值按线性混合方式处理。
- 序列控制仅影响`y`、`output_h`以及后续时间步的递推状态，`update`、`update_att`、`reset`、`new`和`hidden_new`仍保存当前时间步的计算结果。
- 支持编译阶段的动态shape和动态rank，运行时须解析为满足约束的具体shape。
- 内部状态递推使用FP32计算，输出按声明的数据类型写回；输出舍入值不反馈到递推。将序列拆成多次调用时，不保证与一次完整调用逐位相同。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| 图模式调用 | [test_geir_dynamic_augru](examples/arch35/test_geir_dynamic_augru.cpp) | 通过[算子IR](op_graph/dynamic_augru_proto.h)构图方式调用DynamicAUGRU，验证静态shape、动态shape和动态rank。 |

TF 插件注册的 `OriginOpType` 为 `DynamicAUGRU`。该注册用于 TF 图节点到 GE 算子的映射，不能据此推断存在同名 `tf.raw_ops` 或 TF-Adapter Python 函数。本目录提供 GEIR 调用示例，未提供 TF/ONNX 端到端导入示例。重复执行的逐位确定性尚无专项验证结果。
