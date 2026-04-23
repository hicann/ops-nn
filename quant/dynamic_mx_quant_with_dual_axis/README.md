# DynamicMxQuantWithDualAxis

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：在-1轴和-2轴上同时进行目的数据类型为FLOAT4类、FLOAT8类的MX量化。在给定的-1轴和-2轴上，每32个数，计算出这两组数对应的量化尺度mxscale1、mxscale2，然后分别对两组数所有元素除以对应的mxscale1或mxscale2，根据round_mode转换到对应的dst_type，得到量化结果y1和y2。在dst_type为FLOAT8_E4M3FN、FLOAT8_E5M2时，根据scale_alg的取值来指定计算mxscale的不同算法。

- 合轴说明：算子实现时，会对-2轴（不包含）之前的所有轴进行合轴处理。即对于输入shape为$(d_0, d_1, ..., d_{n-3}, d_{n-2}, d_{n-1})$的张量，-2轴之前的维度$(d_0, d_1, ..., d_{n-3})$会被合并为一个维度，等效于将输入reshape为$(d_0 \times d_1 \times ... \times d_{n-3}, d_{n-2}, d_{n-1})$后再进行量化计算。

- 计算公式：
  - 场景1，当scale_alg为0时，即OCP Microscaling Formats (Mx) Specification实现：
  - 将输入x在-1轴上按照32个数进行分组，一组32个数 $\{\{V_i\}_{i=1}^{32}\}$ 量化为 $\{mxscale1, \{P_i\}_{i=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale1 = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32
    $$

  - 同时，将输入x在-2轴上按照32个数进行分组，一组32个数 $\{\{V_j\}_{j=1}^{32}\}$ 量化为 $\{mxscale2, \{P_j\}_{j=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_j(|V_j|))) - emax
    $$

    $$
    mxscale2 = 2^{shared\_exp}
    $$

    $$
    P_j = cast\_to\_dst\_type(V_j/mxscale2, round\_mode), \space j\space from\space 1\space to\space 32
    $$

  - -1轴​量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出y1，mxscale1按对应的-1轴维度上的分组组成输出mxscale1。-2轴​量化后的 $P_{j}$ 按对应的 $V_{j}$ 的位置组成输出y2，mxscale2按对应的-2轴维度上的分组组成输出mxscale2。

  - emax: 对应数据类型的最大正则数的指数位。

    |   DataType    | emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

  - 场景2，当scale_alg为1时，只涉及FP8类型（CuBALS Scale计算算法）：
    - 将输入x在-1轴（或-2轴）上按照32个数进行分组，对每组单独计算块缩放因子$S_{fp32}^b$，再把组内所有元素映射到目标低精度类型FP8。
    - 找到该组中数值的最大绝对值：$Amax(D_{fp32}^b) = max(\{|d_i|\}_{i=1}^{32})$
    - 将FP32映射到目标数据类型FP8可表示的范围内：$S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}$
    - 从块缩放因子中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$，并进行条件向上取整。
    - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$，计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - 对每个组内元素应用量化：$d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$

  - 场景3，当scale_alg为2时，只涉及FP4_E2M1类型（DynamicDtypeRange算法，仅V2接口支持）：
    - 当dstTypeMax = 0.0/6.0/7.0时，使用指数域addValueBit进位法计算scale。
    - 当dstTypeMax为其他自定义值时，使用FP32精度invDstTypeMax乘法法计算scale。

## 参数说明
<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 330px">
  <col style="width: 280px">
  <col style="width: 120px">
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
      <td>待量化数据，对应公式中V<sub>i</sub>和d<sub>i</sub>。<br>目的类型为FLOAT4_E2M1、FLOAT4_E1M2时，x的最后一维必须是偶数。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>可选属性</td>
      <td>数据转换的模式。<br>当dst_type为40/41(FLOAT4_E2M1/FLOAT4_E1M2)时，支持{"rint", "floor", "round"}；<br>当dst_type为35/36(FLOAT8_E5M2/FLOAT8_E4M3FN)时，仅支持{"rint"}；<br>传入空指针时，采用"rint"模式。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>可选属性</td>
      <td>指定数据转换后y1和y2的类型。<br>输入范围为{35, 36, 40, 41}，分别对应{35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2}。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>可选属性</td>
      <td>mxscale1和mxscale2的计算方法。<br>支持取值0、1和2，取值为0代表OCP实现（场景1），为1代表CuBALS实现（场景2），为2代表DynamicDtypeRange实现（场景3）。<br>当dst_type为FLOAT4_E1M2时仅支持取值为0。<br>当dst_type为FLOAT4_E2M1时仅支持取值为0和2。<br>当dst_type为FLOAT8时仅支持取值为0和1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>可选属性</td>
      <td>maxType的取值，对应公式中的Amax(DType)。<br>支持取值0.0和6.0-12.0，取值为0.0代表Amax(DType)为量化结果数据类型的最大值；取值为6.0-12.0代表Amax(DType)为传入值。<br>仅支持在FP4_E2M1和scale_alg为2时设置该值。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>输入x量化-1轴后的对应结果，对应公式中的P<sub>i</sub>和d<sup>i</sup>。<br>shape和输入x一致。</td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale1</td>
      <td>输出</td>
      <td>-1轴每个分组对应的量化尺度，对应公式中的mxscale1和S<sup>b</sup>。<br>shape为x的-1轴的值除以32向上取整，并对其进行偶数pad，pad填充值为0。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>输入x量化-2轴后的对应结果，对应公式中的P<sub>j</sub>和d<sup>j</sup>。<br>shape和输入x一致。</td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale2</td>
      <td>输出</td>
      <td>-2轴每个分组对应的量化尺度，对应公式中的mxscale2和S<sup>b</sup>。<br>shape为x的-2轴的值除以32向上取整，并对其进行偶数pad，pad填充值为0。<br>mxscale2输出需要对每两行数据进行交织处理。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

 - 关于x、mxscale1、mxscale2的shape约束说明如下：
    - x的维度应该大于等于2。
    - rank(mxscale1) = rank(x) + 1。
    - rank(mxscale2) = rank(x) + 1。
    - mxscale1.shape[-2] = (ceil(x.shape[-1] / 32) + 2 - 1) / 2。
    - mxscale2.shape[-3] = (ceil(x.shape[-2] / 32) + 2 - 1) / 2。
    - mxscale1.shape[-1] = 2。
    - mxscale2.shape[-1] = 2。
    - 其他维度与输入x一致。
    - 举例：输入x的shape为[B, M, N]，目的数据类型为FP8类时，对应的y1和y2的shape为[B, M, N]，mxscale1的shape为[B, M, (ceil(N/32)+2-1)/2, 2]，mxscale2的shape为[B, (ceil(M/32)+2-1)/2, N, 2]。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口(V1)  | [test_aclnn_dynamic_mx_quant_with_dual_axis](examples/arch35/test_aclnn_dynamic_mx_quant_with_dual_axis.cpp) | 通过[aclnnDynamicMxQuantWithDualAxis](docs/aclnnDynamicMxQuantWithDualAxis.md)接口方式调用，支持scale_alg=0/1。 |
| aclnn接口(V2)  | [test_aclnn_dynamic_mx_quant_with_dual_axis_v2](examples/arch35/test_aclnn_dynamic_mx_quant_with_dual_axis_v2.cpp) | 通过[aclnnDynamicMxQuantWithDualAxisV2](docs/aclnnDynamicMxQuantWithDualAxisV2.md)接口方式调用，支持scale_alg=0/1/2。 |
| 图模式 | -  | 通过[算子IR](op_graph/dynamic_mx_quant_with_dual_axis_proto.h)构图方式调用DynamicMxQuantWithDualAxis算子。         |
