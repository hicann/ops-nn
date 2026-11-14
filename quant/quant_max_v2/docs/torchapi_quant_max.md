# quant_max

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
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

- 接口功能：`quant_max` 按 `group_list` 给出的分组（`group_list=None` 时整个 tensor 视为单组，即 Per-Tensor）对输入 `x` 求每组 `|x|` 的最大值 `amax`，并输出量化结果 `y`。传入 `scale` 时为静态量化；不传（`None`）时为动态量化，此时额外输出量化系数 `y_scale`。`amax` 恒为必选输出。底层封装 aclnnQuantMaxV2，详细参数契约参见 [aclnnQuantMaxV2](aclnnQuantMaxV2.md)。

- 计算公式：

  `group_list` 沿 dim-0 将 `x` 切分为 `numGroups`（记 `G`）组，`group_list[g]` 为前 `g+1` 组在 dim-0 上的累计结束位置（cumsum 语义）。对每组：

  $$
  a_g = \max\left(|x_g|\right)
  $$

  静态量化（传入 `scale`，`scale[g] = 1/s` 为常规量化系数的倒数）：

  $$
  q_g = scale[g], \qquad y_g = \mathrm{Cast}\left(x_g \times q_g,\ dstType,\ roundMode\right)
  $$

  动态量化（不传 `scale`）：

  $$
  q_g = \begin{cases}
  \dfrac{dstTypeMax}{a_g}, & a_g > 0 \\[4pt]
  0, & a_g = 0
  \end{cases}, \qquad
  y_g = \mathrm{Cast}\left(x_g \times q_g,\ dstType,\ roundMode\right)
  $$

  输出：

  $$
  amax[g] = a_g\ (\text{恒输出}), \qquad
  yScale[g] = \dfrac{1}{q_g}\ (\text{仅动态量化输出}), \qquad
  y = \mathrm{Concat}\left(y_0, \ldots, y_{G-1}\right)
  $$

## 函数原型

```python
cann_ops_nn.quant_max(
    x,
    scale=None,
    group_list=None,
    *,
    round_mode,
    dst_type,
    dst_type_max=15.0,
) -> (Tensor, Tensor, Tensor)
```

也可以直接使用已注册的PyTorch算子：

```python
torch.ops.cann_ops_nn.quant_max(
    x, scale, group_list,
    round_mode=round_mode,
    dst_type=dst_type,
    dst_type_max=dst_type_max,
)
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1501px"><colgroup>
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
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>待量化张量，对应公式中的 <code>x</code>。</td>
        <td>float32、float16、bfloat16</td>
        <td>2-8 维</td>
    </tr>
    <tr>
        <td>scale</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>量化系数 <code>s</code> 的倒数，对应公式中的 <code>scale</code>。<code>None</code> 表示动态量化（系数按本组 <code>amax</code> 现算）；传入表示静态量化。</td>
        <td>float32</td>
        <td><code>[numGroups]</code>（Per-Tensor 为 <code>[1]</code>）</td>
    </tr>
    <tr>
        <td>group_list</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>输入 <code>x</code> 在 dim-0 上各分组的累计结束位置（cumsum 语义），对应公式中的 <code>group_list</code>。<code>None</code> 表示 Per-Tensor。</td>
        <td>int64</td>
        <td><code>[numGroups]</code>，<code>numGroups ∈ [1, 384]</code></td>
    </tr>
    <tr>
        <td>round_mode</td>
        <td>str</td>
        <td>必选</td>
        <td>量化结果转换到目标类型时的舍入模式。<code>dst_type=34</code> 支持 <code>"round"</code>、<code>"hybrid"</code>；<code>dst_type=35/36</code> 仅 <code>"rint"</code>。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_type</td>
        <td>int</td>
        <td>必选</td>
        <td>量化输出类型：<code>34</code> = HIFLOAT8、<code>35</code> = FLOAT8_E5M2、<code>36</code> = FLOAT8_E4M3FN。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_type_max</td>
        <td>float</td>
        <td>可选</td>
        <td>目标类型最大有限值，仅动态量化参与系数计算，取值 <code>0 ~ 32768</code>，默认 <code>15.0</code>。</td>
        <td>-</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## 返回值说明

<table style="undefined;table-layout: fixed; width:1501px"><colgroup>
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
        <td>y</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>量化结果，对应公式中的 <code>y</code>，shape 与 <code>x</code> 一致。<code>dst_type=34</code> 时为承载 HIFLOAT8 位模式的 <code>torch.uint8</code>（PyTorch 无原生 HIFLOAT8 类型）。</td>
        <td>uint8（34）、float8_e5m2（35）、float8_e4m3fn（36）</td>
        <td>与 <code>x</code> 相同</td>
    </tr>
    <tr>
        <td>y_scale</td>
        <td>Tensor</td>
        <td>可选</td>
        <td><code>y</code> 的量化系数（<code>1/q</code>，动态量化下为 <code>amax/dstTypeMax</code>），满足 <code>x ≈ y × y_scale</code>。仅动态量化输出；静态量化返回 shape 为 <code>[0]</code> 的占位张量（不写入）。</td>
        <td>float32</td>
        <td><code>[numGroups]</code>（静态为 <code>[0]</code> 占位）</td>
    </tr>
    <tr>
        <td>amax</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>各组 <code>x</code> 的绝对值最大值，对应公式中的 <code>amax</code>，恒输出。</td>
        <td>与 <code>x</code> 一致</td>
        <td><code>[numGroups]</code>（Per-Tensor 为 <code>[1]</code>）</td>
    </tr>
</tbody>
</table>

## 约束说明

- 输入和输出必须位于 NPU，`scale`/`group_list` 必须与 `x` 同设备。
- `group_list` 的取值（单调不减、位于 `[0, x.shape[0]]`、末值等于 `x.shape[0]`）由调用方保证，与 V1 一致，接口不做同步 D2H 校验；相邻元素相等表示空组。
- `dst_type_max` 仅对 `dst_type=34` 生效；`35/36` 场景在 tiling 内覆盖为类型最大有限值（57344/448）。
- 该接口支持训练、推理场景下使用。
- 支持单算子模式（eager）与 aclgraph 捕获/重放（`torch.npu.graph`）；**不支持 TorchAir（GE）图模式**（`torch.compile(backend="npu")`）——当前交付形态的 vendor 算子信息未提供图编译所需的 kernel-lib 与 op_select_format 字段，GE 图编译会退回 TBE 动态编译并失败。

## 确定性/Batch一致性

默认支持确定性计算。

## 调用说明

- 单算子模式调用（eager）

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    x = (torch.randn((8192, 2048)) * 2).to(torch.bfloat16).npu()

    # 动态 Per-Tensor：不传 scale，按 amax 现算系数，输出 y_scale
    y, y_scale, amax = cann_ops_nn.quant_max(
        x, None, None, round_mode="round", dst_type=34, dst_type_max=15.0)
    print("y:", y.shape, y.dtype)            # [8192, 2048] uint8（HIFLOAT8 位模式）
    print("amax:", amax.shape, amax.dtype)   # [1] bfloat16
    print("y_scale:", y_scale.shape)          # [1] float32

    # 动态 Per-Group：group_list 为 48 组的累计结束行
    group_list = torch.tensor([((i + 1) * 8192) // 48 for i in range(48)], dtype=torch.int64).npu()
    y, y_scale, amax = cann_ops_nn.quant_max(
        x, None, group_list, round_mode="round", dst_type=34, dst_type_max=15.0)
    print("amax:", amax.shape)               # [48] bfloat16
    print("y_scale:", y_scale.shape)          # [48] float32

    # 静态量化：传入 scale（= 1/s），不输出 y_scale（返回 [0] 占位）
    scale = torch.full((48,), 0.125, dtype=torch.float32).npu()
    y, y_scale, amax = cann_ops_nn.quant_max(
        x, scale, group_list, round_mode="round", dst_type=34, dst_type_max=15.0)
    print("static y_scale:", y_scale.shape)   # [0]
    ```

- aclgraph 捕获/重放

    ```python
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        y, y_scale, amax = torch.ops.cann_ops_nn.quant_max(
            x, None, group_list, round_mode="round", dst_type=34, dst_type_max=15.0)
    graph.replay()   # 采集窗口内输出不可读，重放后结果有效
    ```
