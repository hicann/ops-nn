# ThnnFusedLstmCellGrad
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

- 算子功能：LSTMCell中四个门中matmul后剩余计算的反向传播，计算正向输出四个门激活前的值gates、输入cx、偏置b的梯度。

- 计算公式：

**变量定义**

* **输入梯度**：$\delta h_t$ (`gradHy`)， $\delta c_t$ (`gradC`)
* **前向缓存**：$i，f，g，o$ (各门激活值`storage`)，$c_{t-1}$ (`cx`)，$c_t$ (`cy`)
* **输出梯度**：$\delta a_i，\delta a_f，\delta a_g，\delta a_o$ (存入 `gradInGatesOut`)，$\delta c_{t-1}$ (`gradCPrevOut`)

**第一阶段：中间梯度与状态回传**

首先计算隐藏状态对细胞状态的贡献，并汇总得到当前时刻细胞的总梯度 $\text{grad\_}c_{total}$：

$$
\begin{aligned}
gcx &= \tanh(c_t) \\
\text{grad\_}c_{total} &= \delta h_t \cdot o \cdot (1 - gcx^2) + \delta c_t \\
\delta c_{t-1} &= \text{grad\_}c_{total} \cdot f
\end{aligned}
$$

**第二阶段：门控分量梯度 (Pre-activation)**

根据代码逻辑，各门控在进入激活函数前的梯度 $\delta a$ 计算如下：

$$
\begin{aligned}
\delta a_o &= (\delta h_t \cdot gcx) \cdot o \cdot (1 - o) \\
\delta a_i &= (\text{grad\_}c_{total} \cdot g) \cdot i \cdot (1 - i) \\
\delta a_f &= (\text{grad\_}c_{total} \cdot c_{t-1}) \cdot f \cdot (1 - f) \\
\delta a_g &= (\text{grad\_}c_{total} \cdot i) \cdot (1 - g^2)
\end{aligned}
$$

**第三阶段：参数梯度 (db)**

**1. 偏置梯度 (db)：**对 Batch 维度（$N$）进行求和：

$$
\delta b = \sum_{n=1}^{N} \begin{bmatrix} \delta a_i \\ \delta a_f \\ \delta a_g \\ \delta a_o \end{bmatrix}_n
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
      <td>dhy</td>
      <td>输入</td>
      <td><ul><li>表示LSTMCell正向输出隐藏状态的梯度。</li><li>shape为[batch，hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dc</td>
      <td>输入</td>
      <td><ul><li>表示LSTMCell正向输出细胞状态的梯度。</li><li>shape为[batch，hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cx</td>
      <td>输入</td>
      <td><ul><li>表示LSTMCell正向输入细胞状态。</li><li>shape为[batch，hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cy</td>
      <td>输入</td>
      <td><ul><li>表示LSTMCell正向输出细胞状态。</li><li>shape为[batch，hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>storage</td>
      <td>输入</td>
      <td><ul><li>表示LSTMCell正向输出四个门的激活值。</li><li>shape为[batch, 4 * hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgates</td>
      <td>输出</td>
      <td><ul><li>表示LSTMCell正向中四个门预激活值的梯度。</li><li>shape为[batch, 4 * hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dc_prev</td>
      <td>输入</td>
      <td><ul><li>表示LSTMCell正向中输入细胞状态的梯度。</li><li>shape为[batch，hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>db</td>
      <td>输入</td>
      <td><ul><li>表示LSTMCell正向中输入偏置的梯度。</li><li>shape为[4 * hidden_size]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <td>has_bias</td>
      <td>属性</td>
      <td>是否需要计算bias梯度。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_thnn_fused_lstm_cell_grad.cpp](examples/test_aclnn_thnn_fused_lstm_cell_grad.cpp) | 通过[aclnnThnnFusedLstmCellBackward](docs/aclnnThnnFusedLstmCellBackward.md)接口方式调用ThnnFusedLstmCellGrad算子。 |      |