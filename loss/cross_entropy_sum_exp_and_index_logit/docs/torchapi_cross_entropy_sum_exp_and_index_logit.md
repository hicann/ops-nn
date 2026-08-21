# cross_entropy_sum_exp_and_index_logit

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：本算子为vocab并行（Tensor Parallel）场景下CrossEntropy本地计算段的融合算子。对按vocab维切分后的TP rank本地logits，在`all_reduce(MAX)`之后、`predicted_logits` / `sum_exp_logits`的`all_reduce(SUM)`之前，一次性完成logits平移（减全局最大值）、target越界mask判定、本地target offset计算、target对应logit的gather，以及`exp`与沿vocab维的本地求和，降低小算子launch、GM读写和中间tensor materialization开销。面向超大词表大模型训练场景。

- 计算公式：

  设当前rank本地vocab shard为`vocab_parallel_logits`，全局最大logit为`global_logits_max`，vocab分片范围为`[vocab_start_index, vocab_end_index)`。对每个token `i`、每个本地vocab位置`j`：

  $$
  target\_mask[i] = \begin{cases} 1, & target[i] < vocab\_start\_index \text{ 或 } target[i] \geq vocab\_end\_index \\ 0, & \text{otherwise} \end{cases}
  $$

  $$
  target\_offset[i] = \begin{cases} 0, & target\_mask[i] = 1 \\ target[i] - vocab\_start\_index, & target\_mask[i] = 0 \end{cases}
  $$

  $$
  predicted\_logits[i] = \begin{cases} 0, & target\_mask[i] = 1 \\ vocab\_parallel\_logits[i, target\_offset[i]] - global\_logits\_max[i], & target\_mask[i] = 0 \end{cases}
  $$

  $$
  exp\_logits[i, j] = \exp\big(vocab\_parallel\_logits[i, j] - global\_logits\_max[i]\big)
  $$

  $$
  sum\_exp\_logits[i] = \sum_{j=0}^{V\_local-1} exp\_logits[i, j]
  $$

## 函数原型

```python
cann_ops_nn.cross_entropy_sum_exp_and_index_logit(vocab_parallel_logits, target, global_logits_max, vocab_start_index, vocab_end_index) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

>**说明：**<br>
>
>- 各参数维度含义：N（`prod(target.shape)`）表示展平后的token总数、V_local表示当前rank本地vocab分片长度。二维输入shape为`[N, V_local]`，三维输入shape为`[S, B, V_local]`（此时`N = S * B`）。

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
        <td>vocab_parallel_logits</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>当前TP rank的本地vocab shard logits，公式中的vocab_parallel_logits。支持非连续Tensor，数据格式ND。不支持空Tensor。</td>
        <td>float32、bfloat16</td>
        <td>2维 [N, V_local] 或3维 [S, B, V_local]，shape[:-1] 需与target.shape一致</td>
    </tr>
    <tr>
        <td>target</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>全局vocab索引，公式中的target。支持非连续Tensor，数据格式ND，取值为非负整数。不支持空Tensor。</td>
        <td>int32</td>
        <td>shape与vocab_parallel_logits.shape[:-1] 一致，展平后N范围 [1, 32K]</td>
    </tr>
    <tr>
        <td>global_logits_max</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>all_reduce(MAX) 后得到的全局最大logit，公式中的global_logits_max。支持非连续Tensor，数据格式ND。不支持空Tensor。</td>
        <td>与vocab_parallel_logits一致</td>
        <td>shape与target一致</td>
    </tr>
    <tr>
        <td>vocab_start_index</td>
        <td>int</td>
        <td>必选</td>
        <td>当前rank vocab分片起始索引（全局），公式中的vocab_start_index。需满足vocab_start_index大于等于0</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>vocab_end_index</td>
        <td>int</td>
        <td>必选</td>
        <td>当前rank vocab分片结束索引（全局），公式中的vocab_end_index。需满足vocab_end_index > vocab_start_index且vocab_end_index - vocab_start_index == V_local。</td>
        <td>-</td>
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
        <td>predicted_logits</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>target对应的logit减去global_max的结果，公式中的predicted_logits。target不在当前rank分片内时对应位置为0。</td>
        <td>float32</td>
        <td>同target.shape</td>
    </tr>
    <tr>
        <td>sum_exp_logits</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>本地exp(logits - global_max) 沿最后一维的求和，公式中的sum_exp_logits。</td>
        <td>float32</td>
        <td>同target.shape</td>
    </tr>
    <tr>
        <td>exp_logits</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>exp(vocab_parallel_logits - global_logits_max)，公式中的exp_logits。</td>
        <td>float32</td>
        <td>同vocab_parallel_logits.shape</td>
    </tr>
    <tr>
        <td>target_offset</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>target - vocab_start_index，公式中的target_offset。target不在当前rank分片内时置0。</td>
        <td>int32</td>
        <td>同target.shape</td>
    </tr>
    <tr>
        <td>target_mask</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>vocab越界掩码，1表示target不在当前rank分片内、0表示在内，公式中的target_mask。</td>
        <td>int32</td>
        <td>同target.shape</td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。
- `vocab_parallel_logits`仅支持2维`[N, V_local]`或3维`[S, B, V_local]`，且`shape[:-1]`与`target.shape`、`global_logits_max.shape`完全一致。
- V_local对齐约束：`bfloat16`输入时V_local需为16的倍数，`float32`输入时V_local需为8的倍数（保证UB 32字节对齐）。
- N（`prod(target.shape)`）范围：[1, 32K]；V_local范围：[16, 200K]。
- 算子不支持空Tensor输入。

## 确定性/Batch一致性

- 默认支持确定性计算。
- 默认支持Batch一致性。

## 调用示例

- 单算子模式调用（eager）

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    # 参数设置（二维 [N, V_local]）
    N = 4096
    V_local = 37888
    vocab_start_index = 0
    vocab_end_index = V_local

    # 生成随机数据，并发送到 npu
    vocab_parallel_logits = torch.randn(N, V_local, dtype=torch.bfloat16).npu()
    target = torch.randint(vocab_start_index, vocab_end_index, (N,), dtype=torch.int32).npu()
    # global_logits_max 通常来自跨 rank all_reduce(MAX)，此处用本地 max 模拟
    global_logits_max = vocab_parallel_logits.to(torch.float32).max(dim=-1)[0].to(torch.bfloat16).contiguous()

    # 调用 cross_entropy_sum_exp_and_index_logit 算子
    predicted_logits, sum_exp_logits, exp_logits, target_offset, target_mask = \
        cann_ops_nn.cross_entropy_sum_exp_and_index_logit(
            vocab_parallel_logits,
            target,
            global_logits_max,
            vocab_start_index,
            vocab_end_index,
        )
    ```

- 图模式（torchair）调用

    ```python
    import torch
    import torch_npu
    import torchair
    import cann_ops_nn


    class NetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(
            self,
            vocab_parallel_logits,
            target,
            global_logits_max,
            vocab_start_index,
            vocab_end_index,
        ):
            return cann_ops_nn.cross_entropy_sum_exp_and_index_logit(
                vocab_parallel_logits,
                target,
                global_logits_max,
                vocab_start_index,
                vocab_end_index,
            )


    def cross_entropy_sum_exp_and_index_logit_test():
        # 参数设置（二维 [N, V_local]）
        N = 256
        V_local = 16
        vocab_start_index = 0
        vocab_end_index = V_local

        vocab_parallel_logits = torch.randn(N, V_local, dtype=torch.float32).npu()
        target = torch.randint(vocab_start_index, vocab_end_index, (N,), dtype=torch.int32).npu()
        global_logits_max = vocab_parallel_logits.max(dim=-1)[0].contiguous()

        model = NetModel()
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
        predicted_logits, sum_exp_logits, exp_logits, target_offset, target_mask = model(
            vocab_parallel_logits,
            target,
            global_logits_max,
            vocab_start_index,
            vocab_end_index,
        )


    if __name__ == "__main__":
        cross_entropy_sum_exp_and_index_logit_test()
    ```
