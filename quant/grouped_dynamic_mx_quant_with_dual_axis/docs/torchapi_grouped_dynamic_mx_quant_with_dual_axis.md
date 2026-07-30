# cann_ops_nn.grouped_dynamic_mx_quant_with_dual_axis

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

    根据 x 和 group_index 一次完成双轴 MX 量化，输出 y1、mxscale1、y2、mxscale2。其中 y1、mxscale1 为最后一维方向量化结果，y2、mxscale2 为倒数第二维方向量化结果。

- 计算说明：

    量化以 32 个元素为一个 MX block。scale_alg=1 时，每个 block 生成 FLOAT8_E8M0 类型的 scale，并使用该 scale 的倒数将 block 内元素转换为 dst_type 指定的数据类型。

- 计算公式：

  对一个MX block内的输入数据：

  $$
  D^{b}=[d^{i}]_{i=1}^{k},\quad k=32
  $$

  计算block内元素的最大绝对值：

  $$
  Amax(D^{b})=\max_i(|d^{i}|)
  $$

  当scale_alg=1时，根据目标FP8类型最大可表示值计算FP32 scale：

  $$
  S_{fp32}^{b}=\frac{Amax(D^{b})}{Amax(dst_type)}
  $$

  从块缩放因子$S_{fp32}^{b}$中提取无偏指数$E_{int}^{b}$和尾数$M_{fixp}^{b}$。

  为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：

  $$
  E_{int}^{b} = \begin{cases}
    E_{int}^{b} + 1, & \text{如果} S_{fp32}^{b} \text{为正规数，且} E_{int}^{b} < 254 \text{且} M_{fixp}^{b} > 0 \\
    E_{int}^{b} + 1, & \text{如果} S_{fp32}^{b} \text{为非正规数，且} M_{fixp}^{b} > 0.5 \\
    E_{int}^{b}, & \text{否则}
  \end{cases}
  $$

  计算块缩放因子：

  $$
  S_{ue8m0}^{b}=2^{E_{int}^{b}}
  $$

  计算块转换因子：

  $$
  R_{fp32}^{b}=\frac{1}{fp32(S_{ue8m0}^{b})}
  $$

  对block内每个元素执行量化：

  $$
  p_i=cast\_to\_dst\_type(d^{i}\times R_{fp32}^{b}, round_mode)
  $$

  输出该block对应的scale和量化结果：

  $$
  (S_{ue8m0}^{b},[p_i]_{i=1}^{k})
  $$

## 函数原型

```python
torch.ops.cann_ops_nn.grouped_dynamic_mx_quant_with_dual_axis(
    x, group_index, *, round_mode="rint", scale_alg=1, dst_type=24, dst_type_max=0)
    -> (Tensor y1, Tensor mxscale1, Tensor y2, Tensor mxscale2)
```

## 参数说明

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
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>待量化的输入Tensor，对应公式中的输入数据D。</td>
        <td>torch.float16、torch.bfloat16</td>
        <td>(M, N)</td>
    </tr>
    <tr>
        <td>group_index</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>量化分组索引，采用cumsum形式描述各group边界。G表示分组数；每个元素值需大于0且非递减，最后一个元素需等于M。</td>
        <td>torch.int64</td>
        <td>(G,)</td>
    </tr>
    <tr>
        <td>round_mode</td>
        <td>str</td>
        <td>可选</td>
        <td>量化结果转换到目标输出类型时使用的舍入模式。仅支持"rint"，默认值为"rint"。</td>
        <td>str</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scale_alg</td>
        <td>int</td>
        <td>可选</td>
        <td>scale计算算法，仅支持1，默认值为1。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_type</td>
        <td>int</td>
        <td>可选</td>
        <td>y1和y2的目标输出数据类型。23表示torch.float8_e5m2，24表示torch.float8_e4m3fn，默认值为24。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_type_max</td>
        <td>float</td>
        <td>可选</td>
        <td>预留参数，仅支持0.0，默认值为0.0。</td>
        <td>float</td>
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
        <td>y1</td>
        <td>Tensor</td>
        <td>必选</td>
        <td><code>x</code>量化-1轴后的对应结果，shape与<code>x</code>一致。</td>
        <td>torch.float8_e4m3fn、torch.float8_e5m2</td>
        <td>(M, N)</td>
    </tr>
    <tr>
        <td>mxscale1</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>-1轴每个分组对应的量化尺度。最后一维每2个scale成对存放，N方向按32个元素生成一个scale，进行偶数pad，pad填充值为0。</td>
        <td>torch.uint8（FLOAT8_E8M0）</td>
        <td>(M, ceil(N/64), 2)</td>
    </tr>
    <tr>
        <td>y2</td>
        <td>Tensor</td>
        <td>必选</td>
        <td><code>x</code>量化-2轴后的对应结果，shape与<code>x</code>一致。</td>
        <td>torch.float8_e4m3fn、torch.float8_e5m2</td>
        <td>(M, N)</td>
    </tr>
    <tr>
        <td>mxscale2</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>-2轴每个分组对应的量化尺度。倒数第二维方向按32个元素生成一个scale，进行偶数pad。输出需要对每两行数据进行交织处理。</td>
        <td>torch.uint8（FLOAT8_E8M0）</td>
        <td>(floor(M/64)+G, N, 2)</td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口仅支持训练场景，不支持推理场景。
- 该接口仅支持单算子模式调用，不支持TorchAir图模式调用。
- 数据类型约束：
  - `x`仅支持torch.float16、torch.bfloat16。
  - `group_index`仅支持torch.int64。
  - `y1`和`y2`的数据类型由`dst_type`指定，当前支持torch.float8_e4m3fn（24）和torch.float8_e5m2（23）。
  - `mxscale1`和`mxscale2`的数据类型为torch.uint8（FLOAT8_E8M0）。
- Shape约束：
  - `x`不支持空Tensor，必须为2维，且N需要能被64整除（N % 64 == 0）。
  - `group_index`不支持空Tensor，必须为1维，每个元素值需大于0且非递减，最后一个元素需等于M。
  - `y1`、`y2`的shape需与`x`一致，均为(M, N)。
  - `mxscale1`的shape为(M, ceil(N/64), 2)。
  - `mxscale2`的shape为(floor(M/64)+G, N, 2)，其中G为group_index的元素个数。
- 所有输入Tensor的shape各维度值必须为正数（大于0）。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用（eager）：

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    M = 256
    N = 1024

    # 构造NPU输入
    x = torch.randn(M, N, dtype=torch.bfloat16).npu()
    group_index = torch.tensor([64, 128, 192, 256], dtype=torch.int64).npu()

    # 调用双轴MX量化接口
    y1, mxscale1, y2, mxscale2 = (
        torch.ops.cann_ops_nn.grouped_dynamic_mx_quant_with_dual_axis(
            x,
            group_index,
            round_mode="rint",
            scale_alg=1,
            dst_type=torch.float8_e4m3fn,
            dst_type_max=0.0,
        )
    )
    # 查看四个输出的shape
    print("y1 shape: ", y1.shape)
    print("mxscale1 shape: ", mxscale1.shape)
    print("y2 shape: ", y2.shape)
    print("mxscale2 shape: ", mxscale2.shape)
    ```
