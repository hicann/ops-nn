# SingleLayerLstmGrad
## 产品支持情况

| 产品                                                                            | 是否支持 |
| :------------------------------------------------------------------------------ | :------: |
| <term>昇腾910_95 AI处理器</term>                                                |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                          |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                                          |    ×     |
| <term>Atlas 推理系列产品 </term>                                                 |    ×     |
| <term>Atlas 训练系列产品</term>                                                  |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                                          |    ×     |

## 功能说明

- 算子功能：单层单向LSTM的反向传播，计算正向输入x、权重w、偏置b、初始隐藏状态initH与初始细胞状态initC的梯度。

- 计算公式：

单层LSTM反向传播计算

**前向传播公式**

| 组件 | 公式 |
|------|------|
| 输入拼接 | $\mathbf{z}_t = \begin{bmatrix} \mathbf{h}_{t-1} \\ \mathbf{x}_t \end{bmatrix}$ |
| 遗忘门 | $\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{z}_t + \mathbf{b}_f)$ |
| 输入门 | $\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{z}_t + \mathbf{b}_i)$ |
| 候选状态 | $\mathbf{g}_t = \tanh(\mathbf{W}_g \mathbf{z}_t + \mathbf{b}_c)$ |
| 输出门 | $\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{z}_t + \mathbf{b}_o)$ |
| 细胞状态 | $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t$ |
| 隐藏状态 | $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$ |

其中：
- $\sigma$ 是 sigmoid 函数
- $\odot$ 表示逐元素乘法 (Hadamard product)
- $W_*$ 是可学习的权重矩阵
- $b_*$ 是可学习的偏置项

**反向传播变量定义**

- 总损失：$L = \sum_{t=1}^{T} L_t$
- 隐藏状态梯度：$\delta\mathbf{h}_t = \frac{\partial L}{\partial \mathbf{h}_t}$
- 细胞状态梯度：$\delta\mathbf{c}_t = \frac{\partial L}{\partial \mathbf{c}_t}$

**反向传播算法（时间步 $t \rightarrow t-1$）**

**初始化**

$$
\delta\mathbf{h}_{T} = \mathbf{0}, \quad \delta\mathbf{c}_{T} = \mathbf{0}, \quad \mathbf{f}_{T} = \mathbf{0}
$$

### 循环 $t = T - 1$ 到 $0$

1. **当前隐藏状态梯度**
   
   $$
   \delta\mathbf{h}_t = \frac{\partial L_t}{\partial \mathbf{h}_t} + \delta\mathbf{h}_{\text{next}}
   $$
2. **当前细胞状态梯度**
   
   $$
   \delta\mathbf{c}_t = \delta\mathbf{h}_t \odot \mathbf{o}_t \odot (1 - \tanh^2(\mathbf{c}_t)) + \delta\mathbf{c}_{\text{next}} \odot \mathbf{f}_{\text{next}}
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
   \delta\mathbf{h}_{\text{prev}} = \delta\mathbf{z}_t[1:\dim(\mathbf{h}_{t-1})]
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
   
   $$
   \mathbf{f}_{\text{next}} \leftarrow \mathbf{f}_t
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
\frac{\partial L}{\partial \mathbf{W}_f} = \sum_{t=1}^{T} \delta\mathbf{f}_t \mathbf{z}_t^\top
$$

**LSTM 梯度流动特性**
**长程依赖处理**

$$
\frac{\partial \mathbf{c}_T}{\partial \mathbf{c}_1} = \prod_{k=2}^{T} \mathbf{f}_k \quad \text{(对角矩阵)}
$$

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 250px">
  <col style="width: 200px">
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
      <td><ul><li>表示LSTM输入的序列，公式中的x。</li><li>shape为[T，batch，input_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w</td>
      <td>输入</td>
      <td><ul><li>表示LSTM的权重，对应公式中的W。</li><li>包含wi与wh;shape为[4*hidden_size, input_size+hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>可选输入</td>
      <td><ul><li>表示LSTM的偏置，对应公式中的b。</li><li>shape为[4*hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inith</td>
      <td>输入</td>
      <td><ul><li>表示LSTM的初始hidden状态，对应公式中在t=0时的h(t-1)。</li><li>shape为[1, Batch, hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>initc</td>
      <td>输入</td>
      <td><ul><li>表示LSTM的初始cell状态，对应公式中在t=0时的c(t-1)</li><li>shape为[1, Batch, hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td><ul><li>表示LSTM正向中输出隐藏状态hidden的梯度，对应公式中的δh。</li><li>shape为[T, Batch, hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dh</td>
      <td>输入</td>
      <td><ul><li>表示LSTM正向中输出隐藏状态hidden在T时刻的梯度，对应公式中T时刻的δh(t)。</li><li>shape为[1, Batch, hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dc</td>
      <td>可选输入</td>
      <td><ul><li>表示LSTM正向中输出细胞状态cell的梯度，对应公式中T时刻的δc(t)。</li><li>shape为[1, Batch, hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>i</td>
      <td>输入</td>
      <td><ul><li>表示LSTM正向中输出的输入门的激活值，对应公式中的i。</li><li>shape为[T, Batch, hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>j</td>
      <td>输入</td>
      <td><ul><li>表示LSTM正向中输出的候选cell状态的tanh输出，对应公式中的g。</li><li>shape为[T, Batch, hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>f</td>
      <td>输入</td>
      <td><ul><li>表示LSTM正向中输出的遗忘门的激活值，对应公式中的f。</li><li>shape为[T, Batch, hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>o</td>
      <td>输入</td>
      <td><ul><li>表示LSTM正向中输出的输出门的激活值，对应公式中的o。</li><li>shape为[T, Batch, hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h</td>
      <td>输入</td>
      <td><ul><li>表示LSTM正向中输出的隐藏hidden状态，对应公式中的h。</li><li>shape为[T, Batch, hidden_size]。</li></ul></td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c</td>
      <td>输入</td>
      <td><ul><li>表示LSTM正向中输出的最终cell状态，对应公式中的c。</li><li>shape为[T, Batch, hidden_size]。</li></ul></td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>tanhc</td>
      <td>输入</td>
      <td><ul><li>表示LSTM正向中输出的最终cell状态经过tanh激活函数后的输出，对应公式中的tanh(c)。</li><li>shape为[T, Batch, hidden_size]。</li></ul></td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seq_length</td>
      <td>可选输入</td>
      <td><ul><li>表示实际序列长度对应的掩码矩阵。</li><li>shape为[T, Batch, hidden_size]。</li></td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dw</td>
      <td>输出</td>
      <td><ul><li>表示LSTM输入权重上的梯度，对应公式中的∂L/∂W。</li><li>shape为[4 * hidden_size，input_size+hidden_size]。</li></td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>db</td>
      <td>输出</td>
      <td><ul><li>表示LSTM输入偏置上的梯度，对应公式中的∂L/∂b。</li><li>shape为[4, hidden_size]。</li></td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td><ul><li>表示LSTM输入序列x上的梯度，对应公式中的δx。</li><li>shape为[T, Batch, input_size]。</li></td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dh_prev</td>
      <td>输出</td>
      <td><ul><li>表示LSTM输入inith的梯度，对应公式中在t=0时的δh_prev。</li><li>shape为[1, Batch, hidden_size]。</li></td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>dc_prev</td>
      <td>输出</td>
      <td><ul><li>表示LSTM输入initc的梯度，对应公式中在t=0时的δc_prev。</li><li>shape为[1, Batch, hidden_size]。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>direction</td>
      <td>属性</td>
      <td>LSTM循环迭代方向。仅支持"UNIDIRECTIONAL"与"REDIRECTIONAL"。默认"UNIDIRECTIONAL"表示前向，"REDIRECTIONAL"表示反向。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <td>gate_order</td>
      <td>属性</td>
      <td>ijfo的排布顺序。支持"ijfo"或"ifjo"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_single_layer_lstm_grad.cpp](examples/test_aclnn_single_layer_lstm_grad.cpp) | 通过[aclnnLstmBackward](docs/aclnnLstmBackward.md)接口方式调用SingleLayerLstmGrad算子。 |      |