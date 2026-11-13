# ThnnFusedLstmCell

## 产品支持情况

| 产品                                                                            | 是否支持 |
| :------------------------------------------------------------------------------ | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                                                |     √    |
| <term>Atlas A3系列产品</term>                         |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                                          |    ×     |
| <term>Atlas推理系列产品</term>                                                 |    ×     |
| <term>Atlas训练系列产品</term>                                                  |    ×     |

## 功能说明

- 算子功能：实现长短期记忆网络单元（LSTM Cell）的单步前向计算中，矩阵乘法后的后续计算。

- 计算公式：

  计算门控激活值：

  $$
  \begin{aligned}
  b &= b_{ih} + b_{hh} \\
  gates &= inputGates + hiddenGates + b \\
  i_{out} &= \sigma(gates_{i}) \\
  f_{out} &= \sigma(gates_{f}) \\
  g_{out} &= \tanh(gates_{g}) \\
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
  * 将 $gates$ 沿最后一维平均切分为4个分量，即 $gates \xrightarrow{\text{split}} [gates_i, gates_f, gates_g, gates_o]$
  * 将得到的4个门控激活值沿最后一维拼接成$\text{storage}$，即 $[i_{out}, f_{out}, g_{out}, o_{out}] \xrightarrow{\text{concat}} \text{storage}$
  * $\sigma$ 为Sigmoid激活函数，$\odot$ 为逐元素乘积

## 参数说明

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 130px">
  <col style="width: 144px">
  <col style="width: 400px">
  <col style="width: 200px">
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
      <td>inputGates</td>
      <td>输入</td>
      <td>输入层的4个门，即输入门（Input Gate）、遗忘门（Forget Gate）、候选细胞状态（Cell Candidate）、输出门（Output Gate）的值。非连续Tensor输入由框架自动转换为连续Tensor后执行。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>hiddenGates</td>
      <td>输入</td>
      <td>隐藏层的4个门的值。非连续Tensor输入由框架自动转换为连续Tensor后执行。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cx</td>
      <td>输入</td>
      <td>上一时刻的细胞状态。非连续Tensor输入由框架自动转换为连续Tensor后执行。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputBias</td>
      <td>输入</td>
      <td>输入偏置。可选参数，缺省时等价于全零偏置。非连续Tensor输入由框架自动转换为连续Tensor后执行。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>hiddenBias</td>
      <td>输入</td>
      <td>隐藏层偏置。可选参数，缺省时等价于全零偏置。非连续Tensor输入由框架自动转换为连续Tensor后执行。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>hy</td>
      <td>输出</td>
      <td>当前时刻的隐状态，即当前时刻的输出。非连续Tensor输出由框架自动转换为连续Tensor后执行。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cy</td>
      <td>输出</td>
      <td>当前时刻的细胞状态。非连续Tensor输出由框架自动转换为连续Tensor后执行。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>storage</td>
      <td>输出</td>
      <td>4个门的激活值，提供给反向计算。非连续Tensor输出由框架自动转换为连续Tensor后执行。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
  数据类型不支持BFLOAT16。

## 约束说明

- 输入与输出的数据类型必须一致，仅支持FLOAT、FLOAT16、BFLOAT16。
- inputGates与hiddenGates的shape必须相同且为（B, 4H），cx的shape为（B, H），需满足inputGates.shape[1] == 4 × cx.shape[1]；inputBias与hiddenBias传入时shape为（4H,）。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_thnn_fused_lstm_cell.cpp](examples/test_aclnn_thnn_fused_lstm_cell.cpp) | 通过[aclnnThnnFusedLstmCell](docs/aclnnThnnFusedLstmCell.md)接口方式调用ThnnFusedLstmCell算子。 |
