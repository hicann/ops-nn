# situ_mx_quant

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

- 接口功能：将Situ激活函数与动态MX（Microscaling）量化融合为一个算子，对输入x进行Situ激活后，对激活的结果进行MX量化，输出量化后的结果和scale。底层封装aclnnSituMxQuant。

- 计算公式：

  1. Situ激活：

    $$
    situ_a = \beta \times \tanh(gate / \beta) \times sigmoid(gate)
    $$

    当linear_beta > 0时：

    $$
    up = linear\_beta \times \tanh(up / linear\_beta)
    $$

    $$
    situOut = situ_a \times up
    $$

    其中，当activate_left为true时，gate取x的前半部分，up取后半部分；当activate_left为false时，gate取x的后半部分，up取前半部分。
  2. MX量化（OCP算法）：

    $$
    shared\_exp = floor(log2(max(|situOut_i|))) - emax
    $$

    $$
    y\_scale = 2^{shared\_exp}  (E8M0)
    $$

    $$
    y = cast\_to\_fp8(situOut / y\_scale)
    $$

## 函数原型

```python
cann_ops_nn.situ_mx_quant(
    x,
    beta=1.0,
    linear_beta=0.0,
    activate_left=False,
    dst_type=36,
    round_mode="rint",
) -> (Tensor, Tensor)
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
        <td>输入数据，对应公式中的x。不支持非连续，数据格式支持ND。</td>
        <td>float16、bfloat16</td>
        <td>1-7维，最后一维为2H</td>
    </tr>
    <tr>
        <td>beta</td>
        <td>float</td>
        <td>可选</td>
        <td>Situ激活的beta参数，对应公式中的β。默认1.0。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>linear_beta</td>
        <td>float</td>
        <td>可选</td>
        <td>Situ激活的linear_beta参数，对应公式中的linear_beta。默认0.0。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>activate_left</td>
        <td>bool</td>
        <td>可选</td>
        <td>表示gate取x的前半部分还是后半部分，对应公式中的activate_left。默认false。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_type</td>
        <td>int</td>
        <td>可选</td>
        <td>表示输出y的数据类型，对应公式中cast_to_fp8的目标类型。当前仅支持输入范围为{35, 36}，分别对应输出y的数据类型为{35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}。默认36。</td>
        <td>INT</td>
        <td>-</td>
    </tr>
    <tr>
        <td>round_mode</td>
        <td>str</td>
        <td>可选</td>
        <td>表示量化舍入模式，公式中cast的舍入方式。支持"rint"、"round"、"floor"。默认"rint"。</td>
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
        <td>y</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>量化输出，对应公式中的y。不支持非连续，数据格式支持ND。</td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
        <td>x.shape[:-1]+[H]</td>
    </tr>
    <tr>
        <td>y_scale</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>MX量化的scale（E8M0格式），对应公式中的y_scale。不支持非连续，数据格式支持ND。</td>
        <td>FLOAT8_E8M0</td>
        <td>x.shape[:-1]+[ceil(H/64), 2]</td>
    </tr>
</tbody>
</table>

其中H = x.shape[-1] / 2。

## 约束说明

- x必须为NPU Tensor。
- x的维数必须大于等于1，最后一维必须为偶数。
- x的数据类型必须为torch.float16或torch.bfloat16。
- beta参数必须大于0。
- dst_type支持36(FLOAT8_E4M3FN)或35(FLOAT8_E5M2)。
- round_mode必须为"rint"。
- axis固定为-1（最后一维），暂不支持其他轴。
- 不支持空Tensor和非连续Tensor。
- 关于y_scale的shape约束说明如下：
  - H = x.shape[-1] / 2
  - scaleNum = ceil(H / 64)
  - y_scale.shape = x.shape[:-1] + [scaleNum, 2]
- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用（eager）

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    # FP16 -> FP8_E4M3FN
    x = torch.randn(16, 128, dtype=torch.float16).npu()
    y, y_scale = cann_ops_nn.situ_mx_quant(
        x, beta=1.0, dst_type=36, round_mode="rint",
    )
    print("y:============", y.shape, y.cpu())
    print("y_scale:============", y_scale.shape, y_scale.cpu().view(torch.uint8))

    # BF16 -> FP8_E5M2 + linear_beta
    x_bf16 = torch.randn(32, 256, dtype=torch.bfloat16).npu()
    y_bf16, y_scale_bf16 = cann_ops_nn.situ_mx_quant(
        x_bf16, beta=0.5, linear_beta=2.0, activate_left=False, dst_type=35, round_mode="rint",
    )
    print("y_bf16:============", y_bf16.shape, y_bf16.cpu())
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

        def forward(self, x):
            return cann_ops_nn.situ_mx_quant(
                x, beta=1.0, dst_type=36, round_mode="rint",
            )


    def situ_mx_quant_test():
        x = torch.randn(16, 128, dtype=torch.bfloat16).npu()
        model = NetModel()
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
        y, y_scale = model(x)
        print("y:============", y.shape, y.cpu())
        print("y_scale:============", y_scale.shape, y_scale.cpu().view(torch.uint8))


    if __name__ == "__main__":
        situ_mx_quant_test()
    ```
