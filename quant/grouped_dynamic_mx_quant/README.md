# GroupedDynamicMxQuant

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

- 算子功能：根据传入的分组索引的起始值，对传入的数据进行分组的float8的动态量化。

- 计算公式：
  - 场景1，当scaleAlg为0时：
    - 将输入x在第0维上先按照groupIndex进行分组，每个group内按k = blocksize个数分组，一组k个数 {{x<sub>i</sub>}<sub>i=1</sub><sup>k</sup>} 计算出这组数对应的量化尺度mxscale_pre, {mxscale_pre, {P<sub>i</sub>}<sub>i=1</sub><sup>k</sup>}, 计算公式为下面公式(1)(2)。

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax  \tag{1} 
    $$

    $$
    mxscale\_pre = 2^{shared\_exp}  \tag{2}
    $$

    - 这组数每个数都除以mxscale，根据round_mode转换到对应的dst_type，得到量化结果y，计算公式为下面公式(3)。

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize \tag{3}
    $$

    - ​量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出y，mxscale_pre按对应的groupIndex分组，分组内第一个维度pad为偶数，组成输出mxscale。

    - emax：对应数据类型的最大正则数的指数位。

        |   DataType    | emax |
        | :-----------: | :--: |
        | FLOAT8_E4M3FN |  8   |
        |  FLOAT8_E5M2  |  15  |

  - 场景2，当scaleAlg为1时：
    - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
    - 找到该块中数值的最大绝对值:

      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$

    - 将FP32映射到目标数据类型FP8可表示的范围内，其中$Amax(DType)$是目标精度能表示的最大值。

      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$

    - 将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$
    - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$
    - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：

      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
      $$

    - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
    - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
     - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子，这里指$S_{ue8m0}^b$，$[d^i]_{i=1}^k$代表块内量化后的数据。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1010px"><colgroup>
  <col style="width: 130px">
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
        <td>Device侧的aclTensor，计算公式中的输入x。shape仅支持2维。支持非连续的Tensor，支持空Tensor。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>groupIndex</td>
        <td>输入</td>
        <td>Device侧的aclTensor，量化分组的起始索引。shape仅支持1维。支持非连续的Tensor，不支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>roundMode</td>
        <td>属性</td>
        <td>host侧的string，公式中的round_mode，数据转换的模式，仅支持"rint"模式。</td>
        <td>STRING</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dstType</td>
        <td>属性</td>
        <td>host侧的int64_t，公式中的dst_type，指定数据转换后y的类型，输入范围为{35, 36}，分别对应输出y的数据类型为{35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>blocksize</td>
        <td>属性</td>
        <td>host侧的int64_t，公式中的blocksize，指定每次量化的元素个数，仅支持32。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleAlg</td>
        <td>属性</td>
        <td>host侧的int64_t，指定mxscale计算时采用的算法，仅支持0和1。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dstTypeMax</td>
        <td>属性</td>
        <td>host侧的float32，在scale_alg=2时生效。默认值0.0表示max_type为目标数据类型的最大值，若传入其它数值，则需要按照传入的数值计算mxscale。当前支持取值为0.0/6.0-12.0，只支持在FLOAT4_E2M1场景设置该值。</td>
        <td>FLOAT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>Device侧的aclTensor，公式中的输出y，输入x量化后的对应结果。需与dstType对应，shape仅支持2维，支持空Tensor，shape和输入x一致。</td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>mxscale</td>
        <td>输出</td>
        <td>Device侧的aclTensor，公式中的mxscale_pre组成的输出mxscale，每个分组对应的量化尺度。shape仅支持3维，支持空Tensor。假设x的shape为 [m,n]，groupedIndex的shape为 [g]，则mxscale的shape为 [(m/(blocksize * 2)+g), n, 2]。</td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## 约束说明

- 关于x、groupIndex、y、mxscale的约束说明如下：
  - groupIndex中的值必须非递减，且不能小于0，最后一个元素必须为x第一个维度的长度。
  - $rank(mxscale) = rank(x) + 1$。
  - 假设x的shape为 $[m,n]$，groupedIndex的shape为 $[g]$，则mxscale的shape为 $[(m/(blocksize * 2)+g), n, 2]$。
  - $mxscale.shape[-1] = 2$。
  - 输出y的shape与输入x一致。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_grouped_dynamic_mx_quant](./examples/arch35/test_aclnn_grouped_dynamic_mx_quant.cpp) | 通过[aclnnGroupedDynamicMxQuant](./docs/aclnnGroupedDynamicMxQuant.md)接口方式调用GroupedDynamicMxQuant算子。 |
| aclnn调用 | [test_aclnn_grouped_dynamic_mx_quant_v2](./examples/arch35/test_aclnn_grouped_dynamic_mx_quant.cpp) | 通过[aclnnGroupedDynamicMxQuantV2](./docs/aclnnGroupedDynamicMxQuantV2.md)接口方式调用GroupedDynamicMxQuant算子。 |
