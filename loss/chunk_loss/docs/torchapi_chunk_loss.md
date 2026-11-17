# chunk_loss

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：融合计算语言模型头线性变换与稀疏交叉熵，并返回对隐状态的梯度。输入为一chunk的隐状态`hidden`、词表权重`weight`与标签`labels`，在Device侧按词表维流式计算，无需物化完整logits。输出包括标量`loss`、online状态`max_out`/`sum_out`，以及对`hidden`的梯度`d_h`。本接口不计算`weight`的梯度。调用方按`(M, H)`切块后逐段调用；各段使用同一`alpha`再将`loss`相加，结果与整段一次计算一致。

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

## 函数原型

```python
cann_ops_nn.chunk_loss(hidden, weight, labels, alpha, ignore_index=-100) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

>**说明：**<br>
>
>- `hidden`形状为`[M, H]`，`weight`形状为`[V, H]`，`labels`形状为`[M]`。
>- `alpha`在Device侧按FLOAT32与loss/`dZ`相乘做归一化，要求`alpha > 0`。

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 132px">
<col style="width: 480px">
<col style="width: 330px">
<col style="width: 280px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>hidden</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>一个chunk的隐状态，对应公式中的X。</td>
        <td>float16、bfloat16</td>
        <td>[M, H]</td>
    </tr>
    <tr>
        <td>weight</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>词表权重，对应公式中的W。数据类型须与hidden一致。</td>
        <td>float16、bfloat16</td>
        <td>[V, H]</td>
    </tr>
    <tr>
        <td>labels</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>标签，对应公式中的y。非ignore_index取值须落在[0, V)。</td>
        <td>int64</td>
        <td>[M]</td>
    </tr>
    <tr>
        <td>alpha</td>
        <td>float</td>
        <td>必选</td>
        <td>loss与dZ的缩放系数，须大于0。</td>
        <td>float</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ignore_index</td>
        <td>int</td>
        <td>可选</td>
        <td>忽略标签取值，默认-100。对应位置nll与dZ为0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## 返回值说明

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 132px">
<col style="width: 480px">
<col style="width: 330px">
<col style="width: 280px">
</colgroup>
<thead>
<tr>
    <th>输出名</th>
    <th>输出类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>loss</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>标量交叉熵损失，公式中的loss。</td>
        <td>float32</td>
        <td>[]</td>
    </tr>
    <tr>
        <td>max_out</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>online logsumexp 的max状态。</td>
        <td>float32</td>
        <td>[M]</td>
    </tr>
    <tr>
        <td>sum_out</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>online logsumexp 的sum状态。</td>
        <td>float32</td>
        <td>[M]</td>
    </tr>
    <tr>
        <td>d_h</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>对hidden的梯度，公式中的dH。本接口不计算weight梯度。</td>
        <td>同hidden</td>
        <td>[M, H]</td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。
- `hidden`/`weight`仅支持2维，且最后一维H一致；`labels`为1维，长度等于M。
- 不支持空Tensor输入；不支持语言模型头bias与label smoothing。
- 非`ignore_index`的`labels`取值须落在`[0, V)`，越界行为未定义，由调用方保证。

## 确定性/Batch一致性

- 默认支持确定性计算。
- 默认支持Batch一致性。

## 调用示例

- 单算子模式调用（eager）

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    M, H, V = 128, 256, 1024
    alpha = 1.0 / M
    ignore_index = -100

    hidden = torch.randn(M, H, dtype=torch.float16, device="npu")
    weight = torch.randn(V, H, dtype=torch.float16, device="npu")
    labels = torch.randint(0, V, (M,), dtype=torch.int64, device="npu")

    loss, max_out, sum_out, d_h = cann_ops_nn.chunk_loss(
        hidden, weight, labels, alpha, ignore_index
    )
    print(loss, max_out.shape, sum_out.shape, d_h.shape)
    ```

- 图模式调用（TorchAir）

    ```python
    import torch
    import torch_npu
    import torchair
    import cann_ops_nn

    M, H, V = 128, 256, 1024
    alpha = 1.0 / M

    class Model(torch.nn.Module):
        def forward(self, hidden, weight, labels):
            return cann_ops_nn.chunk_loss(hidden, weight, labels, alpha, -100)

    model = Model().npu()
    compiled = torchair.inference.cache_compile(model)

    hidden = torch.randn(M, H, dtype=torch.float16, device="npu")
    weight = torch.randn(V, H, dtype=torch.float16, device="npu")
    labels = torch.randint(0, V, (M,), dtype=torch.int64, device="npu")
    loss, max_out, sum_out, d_h = compiled(hidden, weight, labels)
    ```
