# ChunkLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     ×    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：融合计算语言模型头线性变换与稀疏交叉熵，并返回对隐状态的梯度。输入为一chunk的隐状态`hidden`、词表权重`weight`与标签`labels`，在Device侧按词表维流式计算，无需物化完整logits。输出包括标量`loss`、online状态`max`/`sum`，以及对`hidden`的梯度`dH`。本算子不计算`weight`的梯度。调用方按`(M, H)`切块后逐段调用；各段使用同一`alpha`再将`loss`相加，结果与整段一次计算一致。本算子不接收chunkSize。
- 计算公式：

  `hidden`即为$X$，形状为$[M, H]$。`labels`即为$y$，形状为$[M]$。`weight`即为$W$，形状为$[V, H]$。

  $$
  Z = X W^{\mathsf T}
  $$

  第$i$个token的负对数似然为：

  $$
  \mathrm{nll}_i =
  \begin{cases}
  0 & y_i = \mathrm{ignore\_index} \\
  \mathrm{logsumexp}(Z_i) - Z_{i, y_i} & 0 \le y_i < V
  \end{cases}
  $$

  $$
  loss = \alpha \sum_{i=0}^{M-1} \mathrm{nll}_i
  $$

  $$
  \mathrm{dZ}_i =
  \begin{cases}
  0 & y_i = \mathrm{ignore\_index} \\
  \alpha\left(\mathrm{softmax}(Z_i) - e_{y_i}\right) & 0 \le y_i < V
  \end{cases}
  $$

  $$
  \mathrm{dH} = \mathrm{dZ}\, W
  $$

  其中：

  - $M$为token数，等于`hidden`的第0维，也等于`labels`的长度。
  - $H$为`hidden`与`weight`的最后一维。
  - $V$为`weight`的第0维，即词表大小。
  - $e_{y_i}$为长度为$V$的one-hot向量：仅第$y_i$个位置为1，其余为0。
  - $\alpha$为属性`alpha`，在Device侧按FLOAT32与loss/`dZ`相乘做归一化（$\alpha>0$）。
  - $Z$、$\mathrm{logsumexp}$、$\mathrm{softmax}$与`loss`在Device侧按FLOAT32累加计算。

## 参数说明

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>hidden</td>
      <td>输入</td>
      <td>一个chunk的隐状态，对应公式中的X，shape为[M, H]。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>语言模型头权重，对应公式中的W，shape为[V, H]；按行存放，不包含bias。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>labels</td>
      <td>输入</td>
      <td>已与hidden对齐的目标token，对应公式中的y，shape为[M]。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>loss</td>
      <td>输出</td>
      <td>本chunk的标量loss，对应公式中的loss；0维标量。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max</td>
      <td>输出</td>
      <td>每行logits的最大值，shape为[M]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum</td>
      <td>输出</td>
      <td>每行对exp(Z - max)求和的结果，shape为[M]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dH</td>
      <td>输出</td>
      <td>loss对hidden的梯度，shape为[M, H]；ignore行对应梯度为0。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>属性</td>
      <td>loss的归一化乘子，对应公式中的alpha；须为有限数且大于0（mean时常取有效token数的倒数，sum时常取1）。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ignore_index</td>
      <td>属性</td>
      <td>labels中需要忽略的取值；与该值相等的label不计入loss，对应dH行为0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- `hidden`、`weight`、`labels`的shape满足：`hidden`为$[M, H]$，`weight`为$[V, H]$，`labels`为$[M]$；且$M$、$H$、$V$均大于等于1。
- `hidden`与`weight`的数据类型须一致，均为FLOAT16或BFLOAT16。
- 归约方式固定为先对全部token的nll求和，再乘以`alpha`。多chunk调用时每段传入同一`alpha`，再将各段`loss`相加。
- 不支持语言模型头bias，不支持label smoothing，不输出`weight`的梯度。
- `labels`中非`ignore_index`的取值须落在$[0, V)$；越界行为未定义，由调用方保证。
- 当前交付实现为arch22（Atlas A2 / ascend910b、ascend910_93）与arch35（Ascend 950 / ascend950）。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_chunk_loss.cpp](examples/test_aclnn_chunk_loss_arch22.cpp) | 通过[aclnnChunkLoss](docs/aclnnChunkLoss.md)接口方式调用ChunkLoss算子（arch22样例）。 |
| aclnn接口  | [arch35/test_aclnn_chunk_loss_arch35.cpp](examples/arch35/test_aclnn_chunk_loss_arch35.cpp) | 通过[aclnnChunkLoss](docs/aclnnChunkLoss.md)接口方式调用ChunkLoss算子（arch35样例）。 |
| PyTorch接口 | [torch_extension](torch_extension) | 通过[chunk_loss](docs/torchapi_chunk_loss.md)接口方式调用ChunkLoss算子。 |
