# SingleLayerLstm

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

仅支持950系列，OpDef配置值为`ascend950`。内部算子执行单层、单方向LSTM；多层、布局转换和物理维度补齐由公共`aclnnLSTM`编排。

公共950系列路径接受FP32、FP16、BF16，支持time-first / batch-first、可选bias、多层和train / eval；要求单向、dropout=0、非packed输入。不支持的组合会返回错误。

下文将上述Ascend 950PR&950DT系列产品、Atlas A3系列产品和Atlas A2系列产品分别简称为950系列、A3系列和A2系列。

## 功能说明

- 算子功能：计算单层、单方向LSTM的输出序列，并输出七类供反向使用的保存状态。

门序为`ifjo`：

```text
[i_hat, f_hat, j_hat, o_hat] = concat(x_t, h_prev) @ w + b_eff
b_eff = b                         # 未传 bias_hh
b_eff = fp32(b) + fp32(bias_hh)    # 传入两份原始 bias 时，在私有 FP32 空间相加
i = sigmoid(i_hat); f = sigmoid(f_hat); j = tanh(j_hat); o = sigmoid(o_hat)
c = f * c_prev + i * j
h = o * tanh(c)
```

实现流程如下：

```text
aclnnLSTM
  → dynamic_rnn/op_api/lstm_single_layer_adapter.h
  → l0op::SingleLayerLstm
  → op_kernel/arch35/single_layer_lstm.cpp
```

kernel依次完成输入投影、全时间轴递推、输出散布。片上布局由host/device共用；权重和状态是否常驻取决于预算，部分形状需要访问GM。长K归约使用分段补偿累加，分块会影响浮点加法顺序。

反向由公共`aclnnLstmBackward → l0op::SingleLayerLstmGrad`完成，消费相同声明类型的保存状态。

**参数说明：**

以下为内部算子`SingleLayerLstm`的参数，公共接口参见`aclnnLSTM`的参数列表。`T`为序列长度，`B`为batch大小，`I/H`为内部张量的物理输入/隐藏维度；公开逻辑维度与补齐约束见下文。FLOAT、FLOAT16、BFLOAT16分别对应FP32、FP16、BF16。

<table>
  <thead>
    <tr><th>参数名</th><th>输入/输出/属性</th><th>描述</th><th>数据类型</th><th>数据格式</th></tr>
  </thead>
  <tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>输入序列，shape为<code>[T, B, I]</code>；内部仅使用time-first布局。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>w</td>
    <td>输入</td>
    <td>融合权重，shape为<code>[I+H, 4H]</code>；行方向先输入权重、后循环权重，列方向按<code>ifjo</code>排列。当SingleLayerLstmGrad使用相同门序时，其<code>[4H, I+H]</code>权重与本参数互为转置。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>b</td>
    <td>输入</td>
    <td>偏置，shape为<code>[4H]</code>。未传<code>bias_hh</code>时是融合偏置；传入<code>bias_hh</code>时是原始输入偏置<code>bias_ih</code>。公共API无bias时提供零偏置，本内部输入仍为必需项。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>init_h</td>
    <td>输入</td>
    <td>初始隐藏状态，shape为<code>[B, H]</code>；不同于SingleLayerLstmGrad的<code>[1, B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>init_c</td>
    <td>输入</td>
    <td>初始细胞状态，shape为<code>[B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>seq_length</td>
    <td>可选输入</td>
    <td>单个最大有效步数（标量或<code>[1]</code>），适用于整个batch；有效值在<code>[0, T]</code>内，<code>t &gt;= seq_length</code>的输出置零；当前取值处理见约束说明。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias_hh</td>
    <td>可选输入</td>
    <td>原始循环偏置，shape为<code>[4H]</code>；与<code>b</code>分别拓宽后在kernel私有FP32空间相加。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>每个时间步的隐藏状态，shape为<code>[T, B, H]</code>；公共输出序列由此切片/转换布局得到。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>output_h</td>
    <td>输出</td>
    <td>保存的每个时间步隐藏状态，shape为<code>[T, B, H]</code>，数值和dtype与<code>y</code>一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>output_c</td>
    <td>输出</td>
    <td>保存的每个时间步细胞状态，shape为<code>[T, B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>i</td>
    <td>输出</td>
    <td>输入门sigmoid激活值，shape为<code>[T, B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>j</td>
    <td>输出</td>
    <td>候选细胞状态tanh激活值，shape为<code>[T, B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>f</td>
    <td>输出</td>
    <td>遗忘门sigmoid激活值，shape为<code>[T, B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>o</td>
    <td>输出</td>
    <td>输出门sigmoid激活值，shape为<code>[T, B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>tanhc</td>
    <td>输出</td>
    <td>每个时间步的<code>tanh(output_c)</code>，shape为<code>[T, B, H]</code>；对应SingleLayerLstmGrad的输入<code>tanhct</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>direction</td>
    <td>可选属性</td>
    <td>仅支持<code>UNIDIRECTIONAL</code>，默认值同此；不支持的值返回错误。</td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gate_order</td>
    <td>可选属性</td>
    <td>仅支持<code>ifjo</code>（输入门、遗忘门、候选状态、输出门），默认值同此。</td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>logical_input_size</td>
    <td>可选属性</td>
    <td>补齐前的输入特征数；默认<code>-1</code>表示取物理维度<code>I</code>，显式值范围为<code>[0, I]</code>。存储shape保持物理维度。</td>
    <td>INT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>logical_hidden_size</td>
    <td>可选属性</td>
    <td>补齐前的隐藏状态特征数；默认<code>-1</code>表示取物理维度<code>H</code>，显式值范围为<code>(0, H]</code>。存储shape保持物理维度。</td>
    <td>INT</td>
    <td>-</td>
  </tr>
  </tbody>
</table>

所有浮点IO使用同一声明类型，整数`seq_length`除外。kernel在私有workspace中拓宽x、w、b，以FP32进行输入投影和递推，写出八个输出时收窄。跨层和跨正反向传递的状态均使用声明dtype。

公共API分别传入两份原始bias，kernel在私有FP32 workspace中相加，避免融合后转回FP16/BF16的额外舍入或溢出。省略bias_hh的内部调用仍使用b作为融合bias；输出及保存状态均使用输入的dtype。

## 约束说明

- 内部张量的`T/B/I/H`均须大于0，物理`I/H`均须为8的倍数；所有浮点IO使用同一dtype及ND格式。公共API负责按门补齐权重和最终切片，内部输出仍保留物理shape。
- 公共逻辑`I=0`仍有非空递推，经适配后使用正的物理输入维度并设置`logical_input_size=0`。公共`B=0/H=0`由适配层处理空输出；内部kernel要求物理`B/H`为正。
- `output_h/output_c/i/j/f/o/tanhc`是完整时间轴上的保存状态，dtype与输入相同。公共API的`h_n/c_n`由适配层提取。
- `seq_length`未提供时按完整`T`执行。当前host实现读取可取得的第一个INT64值；取不到数据、输入为空或值不在`[0,T]`时也按`T`处理。公共非packed ACLNN适配不传此输入；该参数表示统一的时间步长度；Grad的同名参数则为逐元素掩码，两者不能互换。
- 内部仅支持`direction=UNIDIRECTIONAL`、`gate_order=ifjo`。多层、batch-first转换和train/eval编排由公共API负责；不满足tiling或片上资源预算要求的shape会返回错误。

## 调用说明

| 调用方式 | 接口文档 | 说明 |
| --- | --- | --- |
| aclnn接口 | [aclnnLSTM](../dynamic_rnn/docs/aclnnLSTM.md) | 950系列非packed输入路径通过适配层调用SingleLayerLstm；公共参数、形状和限制以接口文档为准。 |
