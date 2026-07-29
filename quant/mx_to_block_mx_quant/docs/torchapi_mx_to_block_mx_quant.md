# cann_ops_nn.mx_to_block_mx_quant

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

- 接口功能：
    将调用 npu_dynamic_mx_quant 量化得到的 FLOAT4 的 Tensor 结合 FLOAT8_E8M0 缩放系数，转换为 FLOAT8 分块量化格式，同时输出 -1 轴和 -2 轴方向的量化尺度。

- 计算公式：

  $$
  \begin{aligned}
  scale_{fp8} &= max(mxscale_{fp4\_block}) / MAX\_OFFSET \\
  offset &= mxscale_{fp4\_block} / scale_{fp8} \\
  x_{fp8} &= x_{fp4} \times offset
  \end{aligned}
  $$

  - 其中 $mxscale_{fp4\_block}$ 是输入 mxscale 提供的 FP8_E8M0 缩放系数；$MAX\_OFFSET$ 是输入和输出数据类型之间的最大偏移量；$x_{fp4}$ 是量化得到的 FLOAT4 张量；$x_{fp8}$ 是转换得到的 FLOAT8 张量。

  - MAX_OFFSET 对照表：

    | 输入     | 输出       | MAX_OFFSET |
    | -------- | ---------- | ---------- |
    | FP4_E2M1 | FP8_E5M2   | 13         |
    | FP4_E1M2 | FP8_E5M2   | 15         |
    | FP4_E2M1 | FP8_E4M3FN | 6          |
    | FP4_E1M2 | FP8_E4M3FN | 8          |

## 函数原型

```python
torch.ops.cann_ops_nn.mx_to_block_mx_quant(x, mxscale, *, dst_type=292, x_type=296)
    -> (Tensor, Tensor, Tensor)
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
        <td>算子输入的量化 Tensor，公式中的 x_fp4。</td>
        <td>FLOAT4_E2M1、FLOAT4_E1M2</td>
        <td>2-3</td>
    </tr>
    <tr>
        <td>mxscale </td>
        <td>Tensor</td>
        <td>必选</td>
        <td>调用 DynamicMxQuant 计算得到的量化尺度，公式中的 mxscale_fp4_block。shape 需满足约束说明中的公式。</td>
        <td>FLOAT8_E8M0</td>
        <td>3-4</td>
    </tr>
    <tr>
        <td>dst_type</td>
        <td>int</td>
        <td>可选</td>
        <td>指定输出 y 的数据类型。当前仅支持输入范围为{291, 292}，分别对应输出 y 的数据类型为 {291: FLOAT8_E5M2, 292: FLOAT8_E4M3FN}。</td>
        <td>INT</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x_type</td>
        <td>int</td>
        <td>可选</td>
        <td>表示输入 x 的真实数据类型。当前仅支持输入范围为{296, 297}，分别对应输入 x 的数据类型为 {296: FLOAT4_E2M1, 297: FLOAT4_E1M2}。</td>
        <td>INT</td>
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
        <td>y</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>量化输出，公式中的 x_fp8，shape 与 x 一致，数据类型由 dst_type 决定。</td>
        <td>FLOAT8_E5M2、FLOAT8_E4M3FN</td>
        <td>2-3</td>
    </tr>
    <tr>
        <td>scale1</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示 -1 轴每个分组对应的量化尺度。shape 需满足约束说明中的公式。</td>
        <td>FLOAT8_E8M0</td>
        <td>3-4</td>
    </tr>
    <tr>
        <td>scale2</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>表示 -2 轴每个分组对应的量化尺度，输出需要对每两行数据进行交织处理。shape 需满足约束说明中的公式。</td>
        <td>FLOAT8_E8M0</td>
        <td>3-4</td>
    </tr>
</tbody>
</table>

## 约束说明

- x 只支持 2 维或 3 维输入，且 -2 轴是 64 的倍数, -1 轴是 2 的倍数。
- 关于 x、mxscale、scale1、scale2 的 shape 约束说明如下：
  - rank(mxscale) = rank(x) + 1。
  - mxscale.shape[-2] = (Ceil(x.shape[-1] / 32) + 2 - 1) / 2。
  - mxscale.shape[-1] = 2。
  - 其它维度与输入 x 一致。
- 关于输出 scale1 的 shape 约束说明如下：
  - rank(scale1) = rank(x) + 1。
  - scale1.shape[-2] = (Ceil(x.shape[-1] / 32) + 2 - 1) / 2。
  - scale1.shape[-1] = 2。
  - 其它维度和输入 x 保持一致。
- 关于输出 scale2 的 shape 约束说明如下：
  - rank(scale2) = rank(x) + 1。
  - scale2.shape[-3] = ((Ceil(x.shape[-2] / 32) + 2 - 1) / 2) * 2 / 2。
  - scale2.shape[-2] = x.shape[-1]。
  - scale2.shape[-1] = 2。
  - 其它维度和输入 x 保持一致。
- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用（eager）

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    x_tmp = torch.randn((64, 64), dtype=torch.bfloat16).npu()
    x, mxscale = torch_npu.npu_dynamic_mx_quant(
        x_tmp,
        axis=-1,
        round_mode="rint",
        dst_type=torch_npu.float4_e2m1fn_x2,
        block_size=32,
        scale_alg=0,
    )
    print("npu_dynamic_mx_quant:============")
    print(x)

    y, scale1, scale2 = torch.ops.cann_ops_nn.mx_to_block_mx_quant(
        x, mxscale, dst_type=torch_npu.float8_e5m2, x_type=torch_npu.float4_e1m2fn_x2
    )
    print("npu_mx_to_block_mx_quant:============")
    print("x:============", x.shape, x.cpu())
    print("mxscale:============", mxscale.shape, mxscale.cpu().view(torch.uint8))
    print("y:============", y.shape, y.cpu())
    print("scale1:============", scale1.shape, scale1.cpu().view(torch.uint8))
    print("scale2:============", scale2.shape, scale2.cpu().view(torch.uint8))
    ```

- 图模式（torchair）调用

    ```python
    import torch
    import torch_npu
    import torchair
    import cann_ops_nn

    npu_backend = "npu"


    class NetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(
            self,
            x,
            mxscale,
            dst_type=torch_npu.float8_e4m3fn,
            x_type=torch_npu.float4_e2m1fn_x2,
        ):
            return torch.ops.cann_ops_nn.mx_to_block_mx_quant(
                x, mxscale, dst_type=dst_type, x_type=x_type
            )


    def mx_to_block_mx_quant_test(x_dtype, dst_type, x_type):
        x = torch.randn((64, 32), dtype=torch.float32).to(dtype=x_dtype)
        x = x.npu()
        mxscale = (
            torch.randn((64, 1, 2), dtype=torch.float32)
            .to(dtype=torch.float8_e8m0fnu)
            .npu()
        )
        model = NetModel()
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
        y, scale1, scale2 = model(x, mxscale, dst_type=dst_type, x_type=x_type)
        print("x:============", x.shape, x.cpu())
        print("mxscale:============", mxscale.shape, mxscale.cpu().view(torch.uint8))
        print("y:============", y.shape, y.cpu())
        print("scale1:============", scale1.shape, scale1.cpu().view(torch.uint8))
        print("scale2:============", scale2.shape, scale2.cpu().view(torch.uint8))


    if __name__ == "__main__":
        mx_to_block_mx_quant_test(
            torch.uint8, torch_npu.float8_e4m3fn, torch_npu.float4_e2m1fn_x2
        )
    ```
