# SingleLayerLstmGrad

## 产品支持情况

| 产品                                                                            | 是否支持 |
| :------------------------------------------------------------------------------ | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                                                |    √    |
| <term>Atlas A3系列产品</term>                          |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                                          |    ×     |
| <term>Atlas推理系列产品</term>                                                 |    ×     |
| <term>Atlas训练系列产品</term>                                                  |    ×     |

下文将上述Ascend 950PR&950DT系列产品、Atlas A3系列产品和Atlas A2系列产品分别简称为950系列、A3系列和A2系列。

## 功能说明

**架构与类型配置：**

- A2系列：OpDef配置值为`ascend910b`，支持FP32/FP16，使用`op_kernel/single_layer_lstm_grad.cpp`。
- A3系列：OpDef配置值为`ascend910_93`，支持FP32/FP16，使用`op_kernel/single_layer_lstm_grad.cpp`。
- 950系列：OpDef配置值为`ascend950`，通过独立`OpAICoreConfig`注册FP32/FP16/BF16，使用`op_kernel/arch35/single_layer_lstm_grad.cpp`。所有浮点输入、保存状态和梯度输出使用同一声明类型。kernel内部将输入拓宽到FP32，并以FP32累加后收窄输出；跨算子缓存使用声明类型。
- 950系列的regbase路径使用`DTYPE_W`。窄类型的大形状路径先在私有workspace中拓宽输入并重算，再调用FP32 matmul/反向路径，最后收窄输出。

950系列的FP16/BF16反向在kernel内用`x/w/init_h/init_c/b`重算FP32门值与状态，再执行反向递推，减少低精度缓存舍入对梯度计算的影响。外部七类缓存及梯度仍为声明dtype；FP32重算结果仅在本次kernel内使用。小形状路径按AIV独立处理批次块；大形状路径使用私有workspace，并在时间步之间同步跨核生成的隐藏状态。

窄类型`b`支持原来的融合bias `[4H]`，也支持按`bias_ih`、`bias_hh`顺序拼接的`[8H]`。公共ACLNN适配使用后者，在kernel内拓宽后相加；`db`始终是`[4H]`，适配层将同一个导数分别映射回两份bias。FP32路径及A2/A3系列使用融合bias `[4H]`。

- 算子功能：单层单向LSTM的反向传播，计算正向输入x、权重w、偏置b、初始隐藏状态initH与初始细胞状态initC的梯度。

- 计算公式：

单层LSTM反向传播计算

**前向传播公式**

| 组件 | 公式 |
|------|------|
| 输入拼接 | $\mathbf{z}_t = \begin{bmatrix} \mathbf{x}_t \\ \mathbf{h}_{t-1} \end{bmatrix}$ |
| 遗忘门 | $\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{z}_t + \mathbf{b}_f)$ |
| 输入门 | $\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{z}_t + \mathbf{b}_i)$ |
| 候选状态 | $\mathbf{g}_t = \tanh(\mathbf{W}_g \mathbf{z}_t + \mathbf{b}_g)$ |
| 输出门 | $\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{z}_t + \mathbf{b}_o)$ |
| 细胞状态 | $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t$ |
| 隐藏状态 | $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$ |

其中：

- $\sigma$是sigmoid函数
- $\odot$表示逐元素乘法(Hadamard product)
- $W_*$是可学习的权重矩阵
- $b_*$是可学习的偏置项

**反向传播变量定义**

- 总损失：标量$L$；调用方分别提供输出序列、末端隐藏状态和末端细胞状态的上游梯度`dy`、`dh`、`dc`。
- 隐藏状态梯度：$\delta\mathbf{h}_t = \frac{\partial L}{\partial \mathbf{h}_t}$
- 细胞状态梯度：$\delta\mathbf{c}_t = \frac{\partial L}{\partial \mathbf{c}_t}$

**反向传播算法（时间步$t \rightarrow t-1$）**

**初始化**

$$
\delta\mathbf{h}_{\text{next}} = \mathrm{dh}, \quad \delta\mathbf{c}_{\text{next}} = \mathrm{dc}
$$

以下以`direction=UNIDIRECTIONAL`的单个batch样本为例，向量采用列向量，时间步从0编号；`init_h/init_c`提供第0步之前的状态。`dh/dc`在公式中省略shape为1的首维。权重和偏置梯度从零开始累加，并对所有batch样本求和。

按$t = T - 1, T - 2, \ldots, 0$的顺序，对每个时间步执行以下步骤：

1. **当前隐藏状态梯度**

   $$
   \delta\mathbf{h}_t = \mathrm{dy}_t + \delta\mathbf{h}_{\text{next}}
   $$
2. **当前细胞状态梯度**

   $$
   \delta\mathbf{c}_t = \delta\mathbf{h}_t \odot \mathbf{o}_t \odot (1 - \tanh^2(\mathbf{c}_t)) + \delta\mathbf{c}_{\text{next}}
   $$
3. **门控梯度计算**

   $$
   \delta\mathbf{o}_t = \delta\mathbf{h}_t \odot \tanh(\mathbf{c}_t) \odot \mathbf{o}_t \odot (1 - \mathbf{o}_t)
   $$

   $$
   \delta\mathbf{g}_t = \delta\mathbf{c}_t \odot \mathbf{i}_t \odot (1 - \mathbf{g}_t^2)
   $$

   $$
   \delta\mathbf{i}_t = \delta\mathbf{c}_t \odot \mathbf{g}_t \odot \mathbf{i}_t \odot (1 - \mathbf{i}_t)
   $$

   $$
   \delta\mathbf{f}_t = \delta\mathbf{c}_t \odot \mathbf{c}_{t-1} \odot \mathbf{f}_t \odot (1 - \mathbf{f}_t)
   $$
4. **参数梯度累加**

   $$
   \frac{\partial L}{\partial \mathbf{W}_f} \mathrel{+}= \delta\mathbf{f}_t \mathbf{z}_t^\top
   $$

   $$
   \frac{\partial L}{\partial \mathbf{b}_f} \mathrel{+}= \delta\mathbf{f}_t
   $$

   $$
   \frac{\partial L}{\partial \mathbf{W}_i} \mathrel{+}= \delta\mathbf{i}_t \mathbf{z}_t^\top
   $$

   $$
   \frac{\partial L}{\partial \mathbf{b}_i} \mathrel{+}= \delta\mathbf{i}_t
   $$

   $$
   \frac{\partial L}{\partial \mathbf{W}_g} \mathrel{+}= \delta\mathbf{g}_t \mathbf{z}_t^\top
   $$

   $$
   \frac{\partial L}{\partial \mathbf{b}_g} \mathrel{+}= \delta\mathbf{g}_t
   $$

   $$
   \frac{\partial L}{\partial \mathbf{W}_o} \mathrel{+}= \delta\mathbf{o}_t \mathbf{z}_t^\top
   $$

   $$
   \frac{\partial L}{\partial \mathbf{b}_o} \mathrel{+}= \delta\mathbf{o}_t
   $$
5. **传播到前一时刻**

   $$

   \delta\mathbf{z}_t = \mathbf{W}_f^\top \delta\mathbf{f}_t + \mathbf{W}_i^\top \delta\mathbf{i}_t + \mathbf{W}_g^\top \delta\mathbf{g}_t + \mathbf{W}_o^\top \delta\mathbf{o}_t
   $$

   $$
   \delta\mathbf{x}_t = \delta\mathbf{z}_t[0:I], \quad
   \delta\mathbf{h}_{\text{prev}} = \delta\mathbf{z}_t[I:I+H]
   $$

   $$
   \delta\mathbf{c}_{\text{prev}} = \delta\mathbf{c}_t \odot \mathbf{f}_t
   $$
6. **更新传播变量**

   $$
   \delta\mathbf{h}_{\text{next}} \leftarrow \delta\mathbf{h}_{\text{prev}}
   $$

   $$
   \delta\mathbf{c}_{\text{next}} \leftarrow \delta\mathbf{c}_{\text{prev}}
   $$

**梯度计算原理**

**细胞状态梯度推导**

$$
\delta\mathbf{c}_t = \frac{\partial L}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{c}_t} + \frac{\partial L}{\partial \mathbf{c}_{t+1}} \frac{\partial \mathbf{c}_{t+1}}{\partial \mathbf{c}_t}
$$

其中：

$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{c}_t} = \mathbf{o}_t \odot (1 - \tanh^2(\mathbf{c}_t))
$$

$$
\frac{\partial \mathbf{c}_{t+1}}{\partial \mathbf{c}_t} = \mathbf{f}_{t+1}
$$

**遗忘门梯度推导**

$$
\delta\mathbf{f}_t = \frac{\partial L}{\partial \mathbf{a}_f^t} = \delta\mathbf{c}_t \odot \mathbf{c}_{t-1} \odot \mathbf{f}_t \odot (1 - \mathbf{f}_t)
$$

**参数梯度推导**

$$
\frac{\partial L}{\partial \mathbf{W}_f} = \sum_{t=0}^{T-1} \delta\mathbf{f}_t \mathbf{z}_t^\top
$$

**LSTM梯度流动特性**
**细胞状态直接连接的梯度（固定门值，不含经隐藏状态反馈的其他路径）**

$$
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_s} = \prod_{k=s+1}^{t} \operatorname{diag}(\mathbf{f}_k), \quad 0 \le s < t < T
$$

**参数说明：**

以下为内部l0op算子`SingleLayerLstmGrad`的参数，公共接口参见`aclnnLstmBackward`的参数列表。
`T`为序列长度，`B`为batch大小，`I`为输入特征数，`H`为隐藏状态特征数。

所有浮点输入、保存状态及梯度输出使用同一dtype。

<table>
  <thead>
    <tr><th>参数名</th><th>输入/输出/属性</th><th>描述</th><th>数据类型</th><th>数据格式</th></tr>
  </thead>
  <tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>输入序列，shape为<code>[T, B, I]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>w</td>
    <td>输入</td>
    <td>融合权重，shape为<code>[4H, I+H]</code>；列方向先输入权重<code>W_ih</code>，后循环权重<code>W_hh</code>，行方向门序由<code>gate_order</code>指定。使用<code>ifjo</code>门序时，与正向SingleLayerLstm的权重互为转置；使用<code>ijfo</code>时还需交换候选状态和遗忘门对应的行块。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>b</td>
    <td>可选输入</td>
    <td>融合偏置，通常为<code>[4H]</code>。950系列的FLOAT16/BFLOAT16还支持<code>[8H]</code>：依次存放两份原始<code>bias_ih</code>、<code>bias_hh</code>，在kernel内部拓宽后相加。省略表示无偏置。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>可选输入</td>
    <td>正向主输出，shape为<code>[T, B, H]</code>。当前公共<code>aclnnLstmBackward</code>适配传空，隐藏状态由<code>h</code>提供。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>init_h</td>
    <td>输入</td>
    <td>正向初始隐藏状态，shape为<code>[1, B, H]</code>；不同于SingleLayerLstm正向内部输入的<code>[B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>init_c</td>
    <td>输入</td>
    <td>正向初始细胞状态，shape为<code>[1, B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>h</td>
    <td>输入</td>
    <td>正向保存的每个时间步隐藏状态，shape为<code>[T, B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>c</td>
    <td>输入</td>
    <td>正向保存的每个时间步细胞状态，shape为<code>[T, B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dy</td>
    <td>输入</td>
    <td>对整段正向输出序列的上游梯度，shape为<code>[T, B, H]</code>。与<code>dh</code>是独立输入，两者在对应时间步累加。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dh</td>
    <td>输入</td>
    <td>对该方向末端隐藏状态的上游梯度，shape为<code>[1, B, H]</code>。不需要此梯度时传同shape的零张量，不可省略。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dc</td>
    <td>输入</td>
    <td>对该方向末端细胞状态的上游梯度，shape为<code>[1, B, H]</code>。不需要此梯度时传同shape的零张量，不可省略。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>i</td>
    <td>输入</td>
    <td>正向保存的输入门sigmoid激活值，shape为<code>[T, B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>j</td>
    <td>输入</td>
    <td>正向保存的候选细胞状态tanh激活值（公式中的g），shape为<code>[T, B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>f</td>
    <td>输入</td>
    <td>正向保存的遗忘门sigmoid激活值，shape为<code>[T, B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>o</td>
    <td>输入</td>
    <td>正向保存的输出门sigmoid激活值，shape为<code>[T, B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>tanhct</td>
    <td>输入</td>
    <td>正向保存的<code>tanh(c)</code>，shape为<code>[T, B, H]</code>；对应SingleLayerLstm的输出<code>tanhc</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>seq_length</td>
    <td>可选输入</td>
    <td>逐时间步的有效位置掩码，shape为<code>[T, B, H]</code>，有效位置为1、无效位置为0。950系列的FLOAT16/BFLOAT16路径不支持传入该掩码；非packed公共调用传空。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dw</td>
    <td>输出</td>
    <td>融合权重梯度，shape为<code>[4H, I+H]</code>；布局、门序与<code>w</code>一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>db</td>
    <td>输出</td>
    <td>有偏置时为融合偏置梯度，shape为<code>[4H]</code>。即使<code>b</code>为<code>[8H]</code>，也只输出一份<code>[4H]</code>；两份原始bias的导数相同，由公共适配分别返回。无偏置时l0op分配占位输出，不作为有效偏置梯度读取。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dx</td>
    <td>输出</td>
    <td>输入序列梯度，shape为<code>[T, B, I]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dh_prev</td>
    <td>输出</td>
    <td>初始隐藏状态<code>init_h</code>的梯度，shape为<code>[1, B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dc_prev</td>
    <td>输出</td>
    <td>初始细胞状态<code>init_c</code>的梯度，shape为<code>[1, B, H]</code>。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>direction</td>
    <td>可选属性</td>
    <td>指定对应正向层沿序列的遍历方向，支持<code>UNIDIRECTIONAL</code>（默认）和<code>REDIRECTIONAL</code>。后者表示反向遍历的单方向层，不表示本算子一次计算双向LSTM，也不表示选择是否求梯度。</td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gate_order</td>
    <td>可选属性</td>
    <td>融合权重和偏置的门排列，支持<code>ijfo</code>（默认）或<code>ifjo</code>；公共<code>aclnnLstmBackward</code>适配传<code>ifjo</code>。</td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  </tbody>
</table>

## 约束说明

- 950系列支持FLOAT（FP32）、FLOAT16（FP16）、BFLOAT16（BF16）；A2/A3系列支持FLOAT、FLOAT16。
- 本算子处理单层、单方向、time-first的内部张量。多层、batch-first转换、各层/方向梯度拆分由公共ACLNN适配负责；调用本算子时须按参数表提供内部shape。
- `x/w/init_h/init_c`、三个上游梯度和七类保存状态必须来自同一个正向配置。950系列窄类型路径会重算状态，保存状态输入仍为必需项，dtype须与输入一致。
- 内部要求`T > 0、B > 0、H > 0`。950系列的regbase路径支持`I = 0`递推；共用matmul路径要求`I > 0`。公共API的`H = 0`行为由适配层处理。
- 950系列的FLOAT16/BFLOAT16不支持`seq_length`掩码；该参数要求逐元素掩码，与正向的INT64长度标量不同。双bias的`[8H]`形式仅适用于950系列。
- 是否可以生成tiling还取决于shape、片上空间及workspace大小；不支持的组合会返回错误。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_single_layer_lstm_grad.cpp](examples/test_aclnn_single_layer_lstm_grad.cpp) | 通过[aclnnLstmBackward](docs/aclnnLstmBackward.md)接口方式调用SingleLayerLstmGrad算子。 |
