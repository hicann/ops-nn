# add_rms_norm_dynamic_quant

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

    融合Add、RMS Normalization与MX动态量化，封装aclnnAddRmsNormDynamicMxQuantV2（内部始终调用V2接口，当x3不传入时行为与aclnnAddRmsNormDynamicMxQuant一致）。将输入x1、x2（可选x3，支持三路残差加法）相加后做RmsNorm归一化（可选加beta偏置），再在输入尾轴上按blocksize=32分组进行动态MX量化，输出量化结果y、加法结果x、块量化尺度mxscale及标准差倒数rstd。

- 计算公式：

  当x3不传入时：

  $$
  x=x_{1}+x_{2}
  $$

  当x3传入时（按括号内顺序累加，匹配残差加法语义）：

  $$
  x=(x_{3}+x_{1})+x_{2}
  $$

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  当scale_alg为0时（OCP实现）：

  - 将RmsNorm输出y在尾轴维度上按k = 32个数分组，一组k个数 $\{V_i\}_{i=1}^{k}$ 动态量化为 $\{mxscale,\{P_i\}_{i=1}^{k}\}$

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \quad i = 1, 2, \ldots, k
    $$

  - emax: 对应数据类型的最大正规数的指数位。

    |   DataType    | emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

  当scale_alg为1时（cuBLAS实现），只涉及FP8类型：

  - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。
  - 找到该块中数值的最大绝对值：

    $$
    Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
    $$

  - 将FP32映射到目标数据类型FP8可表示的范围内：

    $$
    S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
    $$

  - 转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$。
  - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$。
  - 为保证量化时不溢出，对指数进行向上取整：

    $$
    E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
    $$

  - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
  - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
  - 应用到量化的最终步骤：$d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$

## 函数原型

```python
cann_ops_nn.add_rms_norm_dynamic_quant(x1, x2, gamma, beta=None, x3=None, epsilon=1e-6, scale_alg=0, round_mode="rint", dst_type=40, output_rstd=False)
    -> (Tensor y, Tensor x, Tensor mxscale, Tensor rstd)
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
        <td>x1</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>标准化过程中的源数据张量，公式中的x1。</td>
        <td>float16、bfloat16</td>
        <td>(..., D)</td>
    </tr>
    <tr>
        <td>x2</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>标准化过程中的源数据张量，公式中的x2。shape和数据类型需与x1一致。</td>
        <td>float16、bfloat16</td>
        <td>(..., D)</td>
    </tr>
    <tr>
        <td>gamma</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>标准化过程中的权重张量，公式中的gamma。shape需与x1最后一维一致。</td>
        <td>float16、bfloat16、float32</td>
        <td>(D,)</td>
    </tr>
    <tr>
        <td>beta</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>标准化过程中的偏置项，公式中的beta。shape和数据类型需与gamma一致。</td>
        <td>float16、bfloat16、float32</td>
        <td>(D,)</td>
    </tr>
    <tr>
        <td>x3</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>三路残差加法的可选输入，公式中的x3。shape和数据类型需与x1一致；不传入时计算x = x1 + x2。</td>
        <td>float16、bfloat16</td>
        <td>(..., D)</td>
    </tr>
    <tr>
        <td>epsilon</td>
        <td>float</td>
        <td>可选</td>
        <td>用于防止除0错误，公式中的epsilon。默认值1e-6。</td>
        <td>float32</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scale_alg</td>
        <td>int</td>
        <td>可选</td>
        <td>mxscale的计算方法，公式中的scale_alg。取值0表示Open Compute Project(OCP)实现，取值1表示cuBLAS实现（仅FP8输出支持）；dst_type为40/41（FP4）时仅支持取值0。默认值0。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>round_mode</td>
        <td>str</td>
        <td>可选</td>
        <td>数据转换的模式，公式中的round_mode。dst_type为35/36（FP8）时仅支持"rint"；dst_type为40/41（FP4）时支持"rint"、"floor"、"round"。默认值"rint"。</td>
        <td>str</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_type</td>
        <td>int</td>
        <td>可选</td>
        <td>输出y的数据类型枚举值，公式中的DType。取值范围为{35, 36, 40, 41}，分别对应FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2。默认值40。</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>output_rstd</td>
        <td>bool</td>
        <td>可选</td>
        <td>是否输出有效的rstd。为False时rstd为无效占位输出（内容未定义）。默认值False。</td>
        <td>bool</td>
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
        <td>y</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>归一化并量化后的结果，公式中的Pi和di。dst_type为40/41（FP4）时，dtype为uint8，两个FP4值打包为一个字节（每4 bit存一个FP4），shape尾轴为x1尾轴的一半；dst_type为35/36（FP8）时，dtype分别为float8_e5m2、float8_e4m3fn，shape与x1一致。</td>
        <td>uint8（FP4）、float8_e5m2、float8_e4m3fn</td>
        <td>FP4: (..., D/2)；FP8: (..., D)</td>
    </tr>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>x1+x2（+x3）的加法结果，公式中的x。shape和数据类型与x1一致。</td>
        <td>float16、bfloat16</td>
        <td>(..., D)</td>
    </tr>
    <tr>
        <td>mxscale</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每个分组（blocksize=32）对应的量化尺度，公式中的mxscale和Sb，E8M0编码。单算子模式下dtype为float8_e8m0fnu；图模式（torchair）下dtype为uint8，二者字节布局一致，可通过view(torch.uint8)互转。shape为x1.shape[:-1]拼接(ceil(ceil(D/32)/2), 2)，即rank比x1多1，倒数第二维为块数向上取整到偶数后的一半，最后一维为2。</td>
        <td>float8_e8m0fnu、uint8</td>
        <td>(..., ceil(ceil(D/32)/2), 2)</td>
    </tr>
    <tr>
        <td>rstd</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>归一化后标准差的倒数，公式中Rms(x)的倒数。output_rstd为True时输出有效值；为False时为无效占位输出（内容未定义）。shape为x1.shape[:-1]拼接(1,)。</td>
        <td>float32</td>
        <td>(..., 1)</td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口支持单算子模式和TorchAir图模式调用。
- x1维度支持1-7维；x2的shape和数据类型需与x1一致；x3（若传入）的shape和数据类型需与x1一致。
- gamma需为一维Tensor，长度等于x1最后一维，数据类型与x1一致或为float32；beta（若传入）的shape和数据类型需与gamma一致。
- dst_type为40/41（FP4）时，x1尾轴必须为偶数，且scale_alg仅支持0。
- scale_alg为1（cuBLAS实现）时仅支持FP8输出。
- mxscale输出shape约束：rank(mxscale) = rank(x1) + 1，mxscale.shape[-2] = ceil(ceil(x1.shape[-1] / 32) / 2)，mxscale.shape[-1] = 2，其他维度与x1一致。

## 确定性计算

默认确定性实现。

## 调用示例

- 单算子模式调用（eager）

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    rows, hidden = 4, 64
    x1 = torch.randn(rows, hidden, dtype=torch.float16).npu()
    x2 = torch.randn(rows, hidden, dtype=torch.float16).npu()
    gamma = torch.ones(hidden, dtype=torch.float16).npu()

    # 不带可选输入，输出FP8_E4M3FN量化结果
    # y, x, mxscale, rstd = cann_ops_nn.add_rms_norm_dynamic_quant(x1, x2, gamma, dst_type=36)

    # 带可选输入beta、x3（三路残差加法）与rstd输出
    beta = torch.zeros(hidden, dtype=torch.float16).npu()
    x3 = torch.randn(rows, hidden, dtype=torch.float16).npu()
    y, x, mxscale, rstd = cann_ops_nn.add_rms_norm_dynamic_quant(
        x1, x2, gamma, beta=beta, x3=x3, epsilon=1e-6,
        scale_alg=0, round_mode="rint", dst_type=36, output_rstd=True)

    # FP4输出（dst_type=40/41）：y为uint8，两个FP4打包为一个字节，尾轴减半
    # y, x, mxscale, rstd = cann_ops_nn.add_rms_norm_dynamic_quant(
    #     x1, x2, gamma, dst_type=40, round_mode="rint")

    # 单算子模式mxscale为float8_e8m0fnu，图模式为uint8；如需统一按uint8处理：
    # mxscale_u8 = mxscale.view(torch.uint8)

    print("y: ", y)
    print("x: ", x)
    print("mxscale: ", mxscale)
    print("rstd: ", rstd)
    ```

- 图模式（torchair）调用

    ```python
    import torch, torch_npu, cann_ops_nn

    @torch.compile(backend="npu")
    def func(x1, x2, gamma):
        return cann_ops_nn.add_rms_norm_dynamic_quant(x1, x2, gamma, dst_type=36)

    x1 = torch.randn(4, 64, dtype=torch.float16).npu()
    x2 = torch.randn(4, 64, dtype=torch.float16).npu()
    gamma = torch.ones(64, dtype=torch.float16).npu()
    y, x, mxscale, rstd = func(x1, x2, gamma)
    print("y: ", y)
    print("mxscale: ", mxscale)
    ```
