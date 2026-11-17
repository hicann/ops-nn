# ChunkLossGrad

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：计算ChunkLoss的权重梯度。根据隐藏状态、投影权重和正向保存的softmax统计量，重算词表得分，输出经归一化及上游梯度缩放后的`gradWeightOut`。
- 计算公式：

  $$
  \ell_{ij}=\sum_{k=0}^{H-1}h_{ik}W_{jk}
  $$

  $$
  p_{ij}=\frac{\exp(\ell_{ij}-m_i)}{s_i},\qquad
  g_{ij}=\begin{cases}
  0,&y_i=\mathrm{ignoreIndex}\\
  \alpha\left(p_{ij}-\mathbf{1}[j=y_i]\right),&y_i\ne\mathrm{ignoreIndex}
  \end{cases}
  $$

  $$
  D_{jk}=q\sum_{i=0}^{N-1}g_{ij}h_{ik}
  $$

  其中，$N$为当前调用的token数量，$H$为隐藏维度，$V$为词表大小；$h$、$W$、$y$分别对应`hidden`、`weight`和`labels`；$\alpha$对应`alpha`；$m$、$s$分别对应`softmaxMax`和`softmaxSum`；$q$对应`gradOutput`；$D$对应输出`gradWeightOut`。$\mathbf{1}[j=y_i]$在$j=y_i$时为1，否则为0。

  `softmaxMax`和`softmaxSum`分别为对应正向计算中每个token的词表得分最大值及减去该最大值后的指数和：

  $$
  m_i=\max_{0\le j<V}\ell_{ij},\qquad
  s_i=\sum_{j=0}^{V-1}\exp(\ell_{ij}-m_i)
  $$

## 参数说明

<table style="table-layout: fixed; width: 1200px">
<colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 600px">
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
  </tr>
</thead>
<tbody>
  <tr>
    <td>hidden</td>
    <td>输入</td>
    <td>隐藏状态，对应公式中的h，shape为[N,H]。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight</td>
    <td>输入</td>
    <td>投影权重，对应公式中的W，shape为[V,H]。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>labels</td>
    <td>输入</td>
    <td>每个token的正确类别索引，对应公式中的y，shape为[N]。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>alpha</td>
    <td>属性</td>
    <td>全局归一化系数，等于正向损失归一化分母的倒数，对应公式中的α。</td>
    <td>FLOAT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>softmaxMax</td>
    <td>输入</td>
    <td>正向保存的逐token词表得分最大值，对应公式中的m，shape为[N]。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>softmaxSum</td>
    <td>输入</td>
    <td>正向保存的逐token稳定化指数和，对应公式中的s，shape为[N]。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradOutput</td>
    <td>输入</td>
    <td>损失的上游梯度，对应公式中的q，shape为[1]；可为正、负或零，单独求当前损失的梯度时传入1。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ignoreIndex</td>
    <td>属性</td>
    <td>需忽略的标签值，与正向一致，默认-100；匹配token的整行得分梯度为零。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gradWeightOut</td>
    <td>输出</td>
    <td>权重梯度，对应公式中的D，shape为[V,H]，数据类型与weight一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 所有Tensor均为必选参数，采用ND格式，位于同一Device，不支持空Tensor。`hidden`、`weight`和`gradWeightOut`的数据类型须相同。
- N、H、V均为正整数且不超过2147483647，N×H和V×H均不超过⌊(2^63−1)/4⌋。
- 非忽略标签须满足0 ≤ labels[i] < V。
- `alpha`为正向损失归一化分母的倒数，须为有限正数且不超过FLOAT32最大有限值，转换为FLOAT32后仍须大于0。跨chunk调用时共用同一全局`alpha`，由调用方累加各次输出。
- `softmaxMax`须为有限值，`softmaxSum`须为有限正值。
- `gradOutput`的shape为[1]，支持正、负或零值。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| aclnn接口 | [test_aclnn_chunk_loss_backward](examples/test_aclnn_chunk_loss_backward.cpp) | 通过[aclnnChunkLossBackward](docs/aclnnChunkLossBackward.md)接口调用ChunkLossGrad算子。 |
| PyTorch接口 | [接口实现](torch_extension/chunk_loss_grad.py) | 通过`cann_ops_nn.chunk_loss_grad`或`torch.ops.cann_ops_nn.chunk_loss_grad`调用。 |
