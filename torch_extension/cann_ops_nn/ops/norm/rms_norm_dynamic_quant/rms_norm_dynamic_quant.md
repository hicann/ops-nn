# cann_ops_nn.rms_norm_dynamic_quant

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：不支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持
- <term>Kirin X90 处理器系列产品</term>：不支持
- <term>Kirin 9030 处理器系列产品</term>：不支持

## 功能说明

- 接口功能：

    融合 RMS Normalization 与对称动态量化（INT8），封装 aclnnRmsNormDynamicQuant。将输入 x 先做 RMSNorm 归一化（可选加 smooth_scales 缩放和 beta 偏置），再按行做动态 INT8 量化，输出 INT8 量化结果和 FLOAT32 的 scale 因子。

- 计算公式：

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma + beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - 若smooth_scales不输入，则直接对rmsnorm输出做量化：

  $$
   scaleOut=row\_max(abs(y))/max\_val
  $$

  $$
   yOut=round(y/scaleOut)
  $$

  - 若输入smooth_scales，则先做smooth缩放再量化：

  $$
    input = y\cdot smooth\_scales
  $$

  $$
   scaleOut=row\_max(abs(input))/max\_val
  $$

  $$
   yOut=round(input/scaleOut)
  $$

  其中row\_max代表每行求最大值。max\_val在INT8时为127。

## 函数原型

```
torch.ops.cann_ops_nn.rms_norm_dynamic_quant(x, gamma, smooth_scales=None, beta=None, *, epsilon=1e-6, dst_type=2)
    -> (Tensor y, Tensor scale)
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
        <td>标准化过程中的源数据张量，公式中的 x。最后一维须 ≤ 8192。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>(..., D)</td>
    </tr>
    <tr>
        <td>gamma</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>标准化过程中的权重张量，公式中的 gamma。shape 需与 x 最后一维一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>(D,)</td>
    </tr>
    <tr>
        <td>smooth_scales</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>量化过程中使用的 smoothScale 张量，公式中的 smooth_scales。shape 和 dtype 需与 gamma 一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>(D,)</td>
    </tr>
    <tr>
        <td>beta</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>标准化过程中的偏置项，公式中的 beta。shape 和 dtype 需与 gamma 一致。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>(D,)</td>
    </tr>
    <tr>
        <td>epsilon</td>
        <td>float</td>
        <td>可选</td>
        <td>用于防止除 0 错误，公式中的 epsilon，必须大于零。默认值 1e-6。</td>
        <td>FLOAT32</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_type</td>
        <td>int</td>
        <td>可选</td>
        <td>输出 y 的数据类型枚举值。当前仅支持 INT8 量化，默认值 2。</td>
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
        <td>量化输出，公式中的 yOut，shape 与 x 一致</td>
        <td>INT8</td>
        <td>(..., D)</td>
    </tr>
    <tr>
        <td>scale</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>量化 scale 因子，公式中的 scaleOut，shape 为 x 去掉最后一维</td>
        <td>FLOAT32</td>
        <td>(...,)</td>
    </tr>
</tbody>
</table>

## 约束说明

- 输入 x 的最后一维必须小于等于 8192。
- 输入 x 的每一行不能全为零（否则归一化后仍全为零，量化 scale 为 0 导致除法异常）。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用（eager）

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    rows, hidden = 16, 64
    x = torch.randn(rows, hidden, dtype=torch.float16).npu()
    gamma = torch.ones(hidden, dtype=torch.float16).npu()

    y, scale = torch.ops.cann_ops_nn.rms_norm_dynamic_quant(x, gamma)
    print("y: ", y)
    print("scale: ", scale)
    ```

    带可选输入：

    ```python
    smooth = torch.ones(hidden, dtype=torch.float16).npu()
    beta = torch.zeros(hidden, dtype=torch.float16).npu()
    y, scale = torch.ops.cann_ops_nn.rms_norm_dynamic_quant(
        x, gamma, smooth_scales=smooth, beta=beta, epsilon=1e-5)
    ```

- 图模式（torchair）调用

    ```python
    import torch, torch_npu, cann_ops_nn

    @torch.compile(backend="npu")
    def func(x, gamma):
        return torch.ops.cann_ops_nn.rms_norm_dynamic_quant(x, gamma)

    x = torch.randn(4, 64, dtype=torch.float16).npu()
    gamma = torch.ones(64, dtype=torch.float16).npu()
    y, scale = func(x, gamma)
    print("y: ", y)
    print("scale: ", scale)
    ```
