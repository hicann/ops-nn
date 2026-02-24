# ThnnFusedLstmCell
## 产品支持情况

| 产品                                                                            | 是否支持 |
| :------------------------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                                                |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                          |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                                          |    ×     |
| <term>Atlas 推理系列产品</term>                                                 |    ×     |
| <term>Atlas 训练系列产品</term>                                                  |    ×     |

## 功能说明

- 算子功能：实现长短期记忆网络单元（LSTM Cell）的单步前向计算中，矩阵乘法后的后续计算。

- 计算公式：

  计算门控激活值：
  
  $$
  \begin{aligned}
  b &= b_{ih} + b_{hh} \\
  gates &= inputGates + hiddenGates + b \\
  i_{out} &= \sigma(gates_{i}) \\
  g_{out} &= \tanh(gates_{g}) \\
  f_{out} &= \sigma(gates_{f}) \\
  o_{out} &= \sigma(gates_{o})
  \end{aligned}
  $$
  
  更新细胞状态：
  
  $$
  cy = f_{out} \odot cx + i_{out} \odot g_{out}
  $$
  
  更新隐状态：
  
  $$
  \begin{aligned}
  tanhc &= \tanh(cy) \\
  hy &= o_{out} \odot tanhc
  \end{aligned}
  $$
  
  相关符号说明：
  
  * 偏置 $b_{ih} = \text{inputBias}$, $b_{hh} = \text{hiddenBias}$
  * 将 $gates$ 沿最后一维平均切分为 4 个分量，即 $gates \xrightarrow{\text{split}} [gates_i, gates_g, gates_f, gates_o]$
  * 将得到的4个门控激活值沿最后一维拼接成$\text{storage}$，即 $[i_{out}, g_{out}, f_{out}, o_{out}] \xrightarrow{\text{concat}} \text{storage}$
  * $\sigma$ 为 Sigmoid 激活函数，$\odot$ 为逐元素乘积

- **参数说明：**
<table><thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>inputGates</td>
      <td>输入</td>
      <td>输入层的4个门，即输入门（Input Gate）、候选细胞状态（Cell Candidate）、遗忘门（Forget Gate）、输出门（Output Gate）的值。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>hiddenGates</td>
      <td>输入</td>
      <td>隐藏层的4个门的值。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cx</td>
      <td>输入</td>
      <td>上一时的刻细胞状态。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputBias</td>
      <td>输入</td>
      <td>输入偏置。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>hiddenBias</td>
      <td>输入</td>
      <td>隐藏层偏置。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>hy</td>
      <td>输出</td>
      <td>当前时刻的隐状态，即当前时刻的输出。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cy</td>
      <td>输出</td>
      <td>当前时刻的细胞状态。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>storage</td>
      <td>输出</td>
      <td>4个门的激活值，提供给反向计算。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_thnn_fused_lstm_cell.cpp](examples/test_aclnn_thnn_fused_lstm_cell.cpp) | 通过[aclnnThnnFusedLstmCell](docs/aclnnThnnFusedLstmCell.md)接口方式调用ThnnFusedLstmCell算子。 |      |