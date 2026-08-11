# matmul_emu_split_weight

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

  使用双路BF16 GEMM融合模拟FP32精度矩阵乘法。离线将FP32权重拆分为高位BF16与低位残差BF16，推理阶段执行两次BF16 GEMM并做线性组合，激活值全程保持BF16，两路矩阵乘均运行在Cube上。底层封装`aclnnMatmulEmuSplitWeight`。

- 计算公式：

  $$
  result = X \times W_{high} + scale \times (X \times W_{low})
  $$

  其中 $X$ 为BF16激活矩阵，$W_{high}$ 为FP32权重截断BF16得到的高位权重，$W_{low}$ 为FP32权重残差除以scale后截断BF16得到的低位残差权重，$scale$ 为缩放因子（典型值1/256=0.00390625）。

- 权重离线拆分公式：

  $$
  W_{high} = \text{BF16}(W_{fp32})
  $$

  $$
  W_{low} = \text{BF16}\left(\frac{W_{fp32} - W_{high}.float()}{scale}\right)
  $$

## 函数原型

```python
cann_ops_nn.matmul_emu_split_weight(x, w_high, w_low, *, w_low_scale=0.00390625, y_dtype=0)
    -> Tensor
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
        <td>激活矩阵，对应公式中的X。K维度必须与w_high/w_low的K维度一致。</td>
        <td>bfloat16</td>
        <td>(M, K)</td>
    </tr>
    <tr>
        <td>w_high</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>高位权重，对应公式中的W_high。由FP32权重截断BF16得到。</td>
        <td>bfloat16</td>
        <td>(K, N)</td>
    </tr>
    <tr>
        <td>w_low</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>低位残差权重，对应公式中的W_low。由FP32权重残差除以scale后截断BF16得到。shape必须与w_high完全一致。</td>
        <td>bfloat16</td>
        <td>(K, N)</td>
    </tr>
    <tr>
        <td>w_low_scale</td>
        <td>float</td>
        <td>可选</td>
        <td>缩放因子，对应公式中的scale。仅支持1/256（0.00390625）。默认值0.00390625。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>y_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>输出数据类型标识。仅支持0（FP32）。默认值0。</td>
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
        <td>result</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>计算输出矩阵，对应公式中的result。双路BF16 GEMM融合模拟FP32精度的矩阵乘结果。</td>
        <td>float32</td>
        <td>(M, N)</td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口支持推理场景下使用。
- x、w_high、w_low的数据类型必须为bfloat16，且数据格式仅支持ND。
- x、w_high、w_low、y均为2维。
- x的shape为(M, K)，w_high的shape为(K, N)，w_low的shape必须与w_high完全一致。
- x的K维度必须与w_high/w_low的K维度一致。
- M、K、N取值范围为(0, INT32_MAX]。
- w_low_scale仅支持1/256（0.00390625），其他值将报错。
- y_dtype仅支持0（FP32），其他值将报错。
- 该接口本身不接收转置属性，如需转置调用方应在传入前对tensor做permute处理，算子支持非连续Tensor，无需额外做contiguous。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用（eager）

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  M, K, N = 128, 256, 128
  x = torch.randn(M, K, dtype=torch.bfloat16).npu()
  w_high = torch.randn(K, N, dtype=torch.bfloat16).npu()
  w_low = torch.randn(K, N, dtype=torch.bfloat16).npu()

  result = cann_ops_nn.matmul_emu_split_weight(x, w_high, w_low)
  print("result: ", result)
  ```
