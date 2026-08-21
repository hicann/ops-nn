# dequant_situ_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
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

- 接口功能：对输入x执行反量化(Dequant)、Situ激活函数和量化(Quant)的融合计算。底层封装aclnnDequantSituQuant。

- 计算公式：

  1. 根据输入数据类型x的不同，反量化路径不同：

  - INT8路径：

    $$
    dequantOut_i = cast\_to\_float(x_i) \times weight\_scale_i + bias_i
    $$

  - INT32路径：

    $$
    dequantOut_i = cast\_to\_float(x_i) \times weight\_scale_i \times activation\_scale_i + bias_i
    $$

  - BFLOAT16/FLOAT16路径（预反量化）：

    $$
    dequantOut_i = cast\_to\_float(x_i)
    $$
  2. Situ激活：

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

    其中，当activate_left为true时，gate取dequantOut的前半部分，up取后半部分；当activate_left为false时，gate取dequantOut的后半部分，up取前半部分。
  3. 量化：

  - static模式：

    $$
    y_i = trunc(situOut_i / quant\_scale_i + quant\_offset_i)
    $$

  - dynamic模式：

    $$
    scale_i = absmax(situOut_i) / 127
    $$

    $$
    y_i = trunc(situOut_i / scale_i)
    $$

## 函数原型

```python
cann_ops_nn.dequant_situ_quant(
    x,
    *,
    weight_scale=None,
    activation_scale=None,
    bias=None,
    quant_scale=None,
    quant_offset=None,
    group_index=None,
    beta=4.0,
    linear_beta=25.0,
    activate_left=True,
    quant_type="dynamic",
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
        <td>int8、int32、bfloat16、float16</td>
        <td>INT8: 2-8维；其他: 2维，最后一维为2H</td>
    </tr>
    <tr>
        <td>weight_scale</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>反量化weight scale，对应公式中的weight_scale。不支持非连续，数据格式支持ND。</td>
        <td>float32</td>
        <td>1维，shape为(2H,)或(1,)</td>
    </tr>
    <tr>
        <td>activation_scale</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>反量化activation scale，对应公式中的activation_scale。不支持非连续，数据格式支持ND。</td>
        <td>float32</td>
        <td>1维，shape为(1,)</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>反量化bias，对应公式中的bias。不支持非连续，数据格式支持ND。</td>
        <td>float32</td>
        <td>1维，shape为(2H,)或(1,)</td>
    </tr>
    <tr>
        <td>quant_scale</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>量化scale，对应公式中的quant_scale。不支持非连续，数据格式支持ND。</td>
        <td>float32</td>
        <td>1维，shape为(H,)或(1,)</td>
    </tr>
    <tr>
        <td>quant_offset</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>量化offset，对应公式中的quant_offset。不支持非连续，数据格式支持ND。</td>
        <td>float32</td>
        <td>1维，shape为(H,)或(1,)</td>
    </tr>
    <tr>
        <td>group_index</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>MoE group索引。不支持非连续，数据格式支持ND。</td>
        <td>int64</td>
        <td>1维，shape为(K,)</td>
    </tr>
    <tr>
        <td>beta</td>
        <td>float</td>
        <td>可选</td>
        <td>Situ激活的beta参数，对应公式中的β。默认4.0。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>linear_beta</td>
        <td>float</td>
        <td>可选</td>
        <td>Situ激活的linear_beta参数，对应公式中的linear_beta。默认25.0。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>activate_left</td>
        <td>bool</td>
        <td>可选</td>
        <td>表示gate取dequantOut的前半部分还是后半部分，对应公式中的activate_left。默认true。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quant_type</td>
        <td>str</td>
        <td>可选</td>
        <td>量化模式，对应公式中的static/dynamic模式。支持"static"和"dynamic"。默认"dynamic"。</td>
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
        <td>int8</td>
        <td>INT8: x.shape[:-1]+[H]；其他: [M, H]</td>
    </tr>
    <tr>
        <td>y_scale</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>动态量化scale，对应公式中的scale（仅dynamic模式有意义）。不支持非连续，数据格式支持ND。</td>
        <td>float32</td>
        <td>INT8: x.shape[:-1]；其他: [M]</td>
    </tr>
</tbody>
</table>

其中H = x.shape[-1] / 2。

## 约束说明

- x及可选Tensor均需为NPU Tensor；可选Tensor可以传None。
- x的最后一维必须为偶数。当x的数据类型为INT8时，维度≥2维；当x的数据类型为INT32/BF16/FLOAT16时，维度为2维。
- beta参数不能为0。
- 当quant_type为"static"时，quant_scale必须提供。
- 支持空Tensor（当x.size(0) == 0时直接返回空输出）。
- 关于quant_type的约束说明如下：

  | quant_type | 含义 | quant_scale | quant_offset |
  | ---- | ---- | ---- | ---- |
  | "static" | 静态量化 | 必选 | 可选 |
  | "dynamic" | 动态量化 | 可选（作为smoothScale） | 不生效 |

- 各数据类型的输入约束如下：

  | x数据类型 | 必选输入 | 可选输入 | 禁止输入 |
  | ---- | ---- | ---- | ---- |
  | INT8 | x, weight_scale | bias, quant_scale, quant_offset | activation_scale, group_index |
  | INT32 | x, weight_scale, activation_scale | bias, group_index | quant_scale, quant_offset |
  | BF16 | x | — | weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index |
  | FLOAT16 | x | — | weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index |

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

    # INT8, dynamic量化
    x = torch.randint(-127, 127, (16, 64), dtype=torch.int8).npu()
    weight_scale = torch.full((64,), 0.1, dtype=torch.float32).npu()

    y, y_scale = cann_ops_nn.dequant_situ_quant(
        x, weight_scale=weight_scale,
        beta=4.0, linear_beta=25.0, activate_left=True, quant_type="dynamic",
    )
    print("y:============", y.shape, y.cpu())
    print("y_scale:============", y_scale.shape, y_scale.cpu())

    # BF16, 预反量化
    x_bf16 = torch.randn(32, 128, dtype=torch.bfloat16).npu()
    y_bf16, y_scale_bf16 = cann_ops_nn.dequant_situ_quant(
        x_bf16, beta=4.0, linear_beta=25.0, activate_left=True, quant_type="dynamic",
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

        def forward(self, x, weight_scale, quant_scale):
            return cann_ops_nn.dequant_situ_quant(
                x, weight_scale=weight_scale, quant_scale=quant_scale,
                beta=4.0, linear_beta=25.0, activate_left=True, quant_type="static",
            )


    def dequant_situ_quant_test():
        x = torch.randint(-127, 127, (16, 64), dtype=torch.int8).npu()
        weight_scale = torch.full((64,), 0.1, dtype=torch.float32).npu()
        quant_scale = torch.tensor([1.0], dtype=torch.float32).npu()
        model = NetModel()
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
        y, y_scale = model(x, weight_scale, quant_scale)
        print("y:============", y.shape, y.cpu())
        print("y_scale:============", y_scale.shape, y_scale.cpu())


    if __name__ == "__main__":
        dequant_situ_quant_test()
    ```
