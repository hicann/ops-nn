# BlockLstmGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     ×    |
|  <term>Atlas A2系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：统一TF BlockLSTMGrad（V1+V2）反向传播计算。基于正向输入与前向缓存，计算BlockLSTM的输入、权重、偏置及各门控的梯度。
- 计算公式：

  **变量定义**

  * **前向输入**：$x$ (`x` [T,B,I])，$cs_{prev}$ (`csPrev` [B,H])，$h_{prev}$ (`hPrev` [B,H])，$w$ (`w` [I+H,4H])，$b$ (`b` [4H])
  * **前向缓存**：$i, cs, f, o, ci, co, h$ (各门激活值与细胞状态 [T,B,H])
  * **梯度输入**：$\delta cs$ (`csGrad` [T,B,H])，$\delta h$ (`hGrad` [T,B,H])
  * **输出梯度**：$\delta x$ (`xGradOut`)，$\delta cs_{prev}$ (`csPrevGradOut`)，$\delta h_{prev}$ (`hPrevGradOut`)，$\delta w$ (`wGradOut`)，$\delta wci$ (`wciGradOut`)，$\delta wcf$ (`wcfGradOut`)，$\delta wco$ (`wcoGradOut`)，$\delta b$ (`bGradOut`)

  **反向计算过程**

  记 $co_t = \tanh(cs_t)$（即前向缓存 `co`），$W_x$、$W_h$ 分别为 $w$ 的前 $I$ 行与后 $H$ 行。沿时间步从 $t=T-1$ 向 $t=0$ 反向递归，逐步回传细胞状态梯度与隐藏状态梯度：

  $$
  \begin{aligned}
  \delta cs_t^{tot} &= \delta h_t \cdot o_t \cdot (1 - co_t^2) + \delta cs_t + \delta cs_{t+1}^{tot} \cdot f_{t+1} \\
  \delta a_i &= \delta cs_t^{tot} \cdot ci_t \cdot i_t \cdot (1 - i_t) \\
  \delta a_f &= \delta cs_t^{tot} \cdot cs_{t-1} \cdot f_t \cdot (1 - f_t) \\
  \delta a_g &= \delta cs_t^{tot} \cdot i_t \cdot (1 - ci_t^2) \\
  \delta a_o &= \delta h_t \cdot co_t \cdot o_t \cdot (1 - o_t)
  \end{aligned}
  $$

  边界：$t=T-1$ 时无 $\delta cs_{t+1}^{tot}$ 项（视其为 0）；$t=0$ 时 $cs_{-1} = cs_{prev}$。

  各时间步的输出与参数梯度（$\delta a$ 为上述四个预激活梯度）：

  $$
  \begin{aligned}
  \delta x_t &= \sum_{gate \in \{i,f,g,o\}} \delta a_{gate} \cdot W_x[gate] \\
  \delta h_{t-1} &= \sum_{gate \in \{i,f,g,o\}} \delta a_{gate} \cdot W_h[gate] \quad (t=0 时即 \delta h_{prev}) \\
  \delta cs_{prev} &= \delta cs_0^{tot} \cdot f_0 \\
  \delta b &= \sum_{t,b} \delta a, \quad \delta w = \sum_{t,b} \begin{bmatrix} x_t \\ h_{t-1} \end{bmatrix} \cdot \delta a
  \end{aligned}
  $$

  其中 $\delta h_{t-1}$ 在 $t>0$ 时并入下一时间步（即时间步 $t-1$）的 $\delta h$ 继续回传。use_peephole=true 时，各门预激活额外包含 peephole 项（$wci \odot cs_{t-1}$、$wcf \odot cs_{t-1}$、$wco \odot co_t$），$\delta wci/\delta wcf/\delta wco$ 按相同方式对 $(t,b)$ 累加。

  $\delta w$ 和 $\delta b$ 的实现对时间步和Batch维度先按 cluster 写入 workspace partial，再以固定顺序归约求和（确定性结果）。

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
      <td>seq_len_max</td>
      <td>输入</td>
      <td>表示序列最大长度。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示BlockLSTM正向输入。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cs_prev</td>
      <td>输入</td>
      <td>表示BlockLSTM正向输入细胞状态初始值。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h_prev</td>
      <td>输入</td>
      <td>表示BlockLSTM正向输入隐藏状态初始值。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w</td>
      <td>输入</td>
      <td>表示BlockLSTM正向权重矩阵。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wci</td>
      <td>输入</td>
      <td>表示peephole权重ci。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wcf</td>
      <td>输入</td>
      <td>表示peephole权重cf。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wco</td>
      <td>输入</td>
      <td>表示peephole权重co。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>输入</td>
      <td>表示BlockLSTM正向偏置。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>i</td>
      <td>输入</td>
      <td>表示BlockLSTM正向输入门激活值。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cs</td>
      <td>输入</td>
      <td>表示BlockLSTM正向细胞状态。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>f</td>
      <td>输入</td>
      <td>表示BlockLSTM正向遗忘门激活值。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>o</td>
      <td>输入</td>
      <td>表示BlockLSTM正向输出门激活值。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ci</td>
      <td>输入</td>
      <td>表示BlockLSTM正向候选状态。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>co</td>
      <td>输入</td>
      <td>表示BlockLSTM正向tanh(cs)值。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h</td>
      <td>输入</td>
      <td>表示BlockLSTM正向隐藏状态。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cs_grad</td>
      <td>输入</td>
      <td>表示BlockLSTM正向输出细胞状态的梯度。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h_grad</td>
      <td>输入</td>
      <td>表示BlockLSTM正向输出隐藏状态的梯度。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_peephole</td>
      <td>属性</td>
      <td>是否使用peephole。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gate_order</td>
      <td>属性</td>
      <td>门控顺序。支持"ifco"和"icfo"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x_grad</td>
      <td>输出</td>
      <td>表示输入x的梯度。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cs_prev_grad</td>
      <td>输出</td>
      <td>表示输入细胞状态初始值的梯度。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h_prev_grad</td>
      <td>输出</td>
      <td>表示输入隐藏状态初始值的梯度。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w_grad</td>
      <td>输出</td>
      <td>表示权重矩阵w的梯度。由确定性归约直接写入，无需调用方清零初始化。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wci_grad</td>
      <td>输出</td>
      <td>表示peephole权重ci的梯度。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wcf_grad</td>
      <td>输出</td>
      <td>表示peephole权重cf的梯度。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wco_grad</td>
      <td>输出</td>
      <td>表示peephole权重co的梯度。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b_grad</td>
      <td>输出</td>
      <td>表示偏置b的梯度。由确定性归约直接写入，无需调用方清零初始化。</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 所有输入与输出参数的数据类型必须为FLOAT或FLOAT16，且需保持一致。
- 仅支持Ascend 950PR&950DT系列产品。
- w_grad和b_grad由确定性归约直接写入，无需调用方清零初始化。
- gate_order支持"ifco"（默认）和"icfo"两种门控顺序。
- seq_len_max须满足 0 <= seq_len_max <= T。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| GEIR调用 | [test_geir_block_lstm_grad](examples/arch35/test_geir_block_lstm_grad.cpp) | 通过GEIR图模式调用BlockLstmGrad算子（固定shape）。 |
| GEIR动态shape调用 | [test_geir_block_lstm_grad_dynamic](examples/arch35/test_geir_block_lstm_grad_dynamic.cpp) | 动态shape（-1）与动态rank（-2）场景，同一图多组shape运行并做双精度golden校验。 |
