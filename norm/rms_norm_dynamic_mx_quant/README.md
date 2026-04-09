# RmsNormDynamicMxQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | × |
|  <term>Kirin 9030 处理器系列产品</term> | × |

## 功能说明

- 接口功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。DynamicMxQuant算子则是在尾轴上按blocksize=32分组进行动态MX量化的算子。RmsNormDynamicMxQuant算子将RmsNorm归一化输出给到的DynamicMxQuant算子融合起来，减少搬入搬出操作。 在输入尾轴axis上，根据每blocksize=32个数，计算出这组数对应的量化尺度mxscale，然后对这组数每一个除以mxscale，根据round_mode转换到对应的dst_type，得到量化结果y。在dst_type为FLOAT8_E4M3FN、FLOAT8_E5M2时，根据scale_alg的取值来指定计算mxscale的不同算法。
- 计算公式：

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  当scale_alg为0时：
   - 将RmsNorm输出y在尾轴维度上按k = 32个数分组，一组k个数 $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale,\{P_i\}_{i=1}^{k}\}$
    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax 
    $$

    $$
    mxscale = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
    $$
    - emax: 对应数据类型的最大正则数的指数位。

        |   DataType    | emax |
        | :-----------: | :--: |
        |  FLOAT4_E2M1  |  2   |
        |  FLOAT4_E1M2  |  0   |
        | FLOAT8_E4M3FN |  8   |
        |  FLOAT8_E5M2  |  15  |

  当scale_alg为1时，只涉及FP8类型：
    - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。
    - 找到该块中数值的最大绝对值:
      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$
    - 将FP32映射到目标数据类型FP8可表示的范围内:
      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$
    - 转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$
    - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$
    - 为保证量化时不溢出，对指数进行向上取整:
      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
      $$
    - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
    - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - 应用到量化的最终步骤：$d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$
    
## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量，对应公式中的x。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示标准化过程中的权重张量，对应公式中的gamma。shape需要与x最后一维一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>可选输入</td>
      <td>表示标准化过程中的偏置项，对应公式中的beta。shape必须与gamma一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>表示添加到分母中的值，以确保数值稳定。对应公式中的epsilon。</li><li>默认值为1e-6。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>可选属性</td>
      <td><ul><li>表示mxscale的计算方法，对应公式中的scale_alg。</li><li>支持取值0和1，取值为0表示Open Compute Project(OCP)实现，取值为1表示cuBLAS实现。当dst_type为FLOAT4_E2M1/FLOAT4_E1M2时仅支持取值为0。</li><li>默认值为0。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示数据转换的模式，对应公式中的round_mode。</li><li>当dst_type为40/41时，支持{"rint", "floor", "round"}。</li><li>当dst_type为36/35时，仅支持{"rint"}。</li><li>默认值为"rint"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>可选属性</td>
      <td><ul><li>表示指定数据转换后y的类型，对应公式中的DType。</li><li>输入范围为{35, 36, 40, 41}，分别对应{FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2}。</li><li>默认值为40。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_rstd</td>
      <td>可选属性</td>
      <td><ul><li>表示指定是否输出有效的rstd。</li><li>支持True和False。</li><li>默认值为False。</li><li>当output_rstd为False时，算子不写出rstd，此时rstd为无效占位输出。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>表示归一化并量化后的结果，对应公式中的Pi和di，shape与x一致。</li></ul></td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale</td>
      <td>输出</td>
      <td><ul><li>表示每个分组对应的量化尺度，对应公式中的mxscale和Sb，shape见约束说明。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输出</td>
      <td><ul><li>表示归一化后的标准差的倒数，对应公式中Rms(x)的倒数。</li><li>shape为与入参x的shape前几维保持一致，前几维指x的维度减去gamma的维度，表示不需要norm的维度。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- <term>Ascend 950PR/Ascend 950DT</term>：
  
  mxscale的shape约束说明如下：
  - rank(mxscale) = rank(x) + 1。
  - mxscale.shape[-2] = (ceil(x.shape[-1] / 32) + 2 - 1) / 2。
  - mxscale.shape[-1] = 2。
  - 其他维度与输入x一致。

- 当输出y的数据类型为FLOAT4_E2M1或FLOAT4_E1M2，x尾轴的值必须为偶数。

- 输入gamma、可选输入beta的数据类型只能和x的数据类型保持一致或者为FLOAT32。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_rms_norm_dynamic_mx_quant](examples/test_aclnn_rms_norm_dynamic_mx_quant.cpp) | 通过[aclnnRmsNormDynamicMxQuant](docs/aclnnRmsNormDynamicMxQuant.md)接口方式调用RmsNormDynamicMxQuant算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/rms_norm_dynamic_mx_quant_proto.h)构图方式调用RmsNormDynamicMxQuant算子。         |
