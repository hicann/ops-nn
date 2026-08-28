# QuantMatmulActivationQuant

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|<term>Ascend 950PR/Ascend 950DT</term>|√|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|×|
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|×|
|<term>Atlas 200I/500 A2 推理产品</term>|×|
|<term>Atlas 推理系列产品</term>|×|
|<term>Atlas 训练系列产品</term>|×|

## 功能说明

- 算子功能：融合量化的矩阵乘、激活以及动态量化，当前支持激活为gelu、MX [量化模式](../../docs/zh/context/quant_mode_introduction.md)。支持FP8（FLOAT8_E4M3FN/FLOAT8_E5M2）和FP4（FLOAT4_E2M1）数据类型输入输出。x2支持ND和NZ两种数据格式，当x2为NZ格式时通过[aclnnQuantMatmulActivationQuantWeightNz](docs/aclnnQuantMatmulActivationQuantWeightNz.md)接口调用，当x2为ND格式时通过[aclnnQuantMatmulActivationQuant](docs/aclnnQuantMatmulActivationQuant.md)接口调用。

- 计算公式：

  - <term>Ascend 950PR/Ascend 950DT</term>：

    - QuantMatmul MX量化模式：

      $$
      matmulOut[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice))* (x1Scale[m/gsM, j] * x2Scale[j, n/gsN]))+bias[n]
      $$

      其中，gsM，gsN和gsK分别代表groupSizeM，groupSizeN和groupSizeK；x1Slice代表x1第m行长度为groupSizeK的向量，x2Slice代表x2第n列长度为groupSizeK的向量；K轴均从j*groupSizeK起始切片，j的取值范围为[0, kLoops)，kLoops = ceil(K / groupSizeK)，K为K轴长度，支持最后的切片长度不足groupSizeK。

    - 激活计算公式：

      - gelu_tanh(高性能近似)：
      $$
      activationOut=GELU(matmulOut)=matmulOut × Φ(matmulOut)=0.5 * matmulOut * (1 + tanh( \sqrt{2 / \pi} * (matmulOut + 0.044715 * matmulOut^{3})))
      $$

      - gelu_erf：
      $$
      activationOut=GELU(matmulOut)=0.5 * matmulOut * (1 + erf(matmulOut / \sqrt{2}))
      $$

    - 动态量化计算公式：

      - 场景1，当scaleAlg为0时：
        - 将输入activationOut在尾轴上按$k = 32$个数分组，一组k个数 $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\},\space k = 32$

        $$
        shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
        mxscale = 2^{shared\_exp}\\
        P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space k\\
        $$

        - 量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出yOut，mxscale按尾轴上的分组输出yScaleOut。

        - emax: 对应数据类型的最大正则数的指数位。

            |   DataType    | emax |
            | :-----------: | :--: |
            |  FLOAT4_E2M1  |  2   |
            |  FLOAT4_E1M2  |  0   |
            | FLOAT8_E4M3FN |  8   |
            |  FLOAT8_E5M2  |  15  |

      - 场景2，当scaleAlg为1时，只涉及FP8类型：
        - 将输入activationOut在尾轴上按$k = 32$个数分块，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。如果最后一块不足$k = 32$个元素，把缺失值视为0，按照完整块处理。
        - 找到该块中数值的最大绝对值:

          $$
          Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
          $$

        - 将FP32映射到目标数据类型FP8可表示的范围内，其中$Amax(DType)$是目标精度能表示的最大值:

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

      - 场景3，当scaleAlg为2时，只涉及FP4_E2M1类型：
        - 当dstTypeMax = 0.0/6.0/7.0时：
          - 将输入activationOut在尾轴上按$k = blocksize$个数分组，一组k个数 $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = blocksize：

          $$
          shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{如果尾数位的高比特前一/两位为1，且尾数不全为0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{其它} \end{cases}
          $$

          $$
          P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize
          $$

          - 量化后的$P_{i}$按对应的$V_{i}$的位置组成输出yOut，mxscale按尾轴上的分组输出yScaleOut。

        - 当dstTypeMax != 0.0/6.0/7.0时：
          - 将输入activationOut在尾轴上按$k = blocksize$个数分块，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型。如果最后一块不足$k$个元素，把缺失值视为0，按照完整块处理。
          - 找到该块中数值的最大绝对值:

            $$
            Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
            $$

          - 将FP32映射到目标数据类型可表示的范围内，其中当dstTypeMax=0时，$Amax(DType)$是目标精度能表示的最大值；当dstTypeMax!=0时，$Amax(DType)$是dstTypeMax传入值。

            $$
            S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
            $$

          - 将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$。
          - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$。
          - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：

            $$
            E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b, & \text{否则} \end{cases}
            $$

          - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
          - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
          - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子，这里指$S_{ue8m0}^b$，$[d^i]_{i=1}^k$代表块内量化后的数据。
          - 量化后的$P_{i}$按对应的$V_{i}$的位置组成输出yOut，mxscale按尾轴上的分组输出yScaleOut。

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 1166px"><colgroup>
<col style="width: 81px">
<col style="width: 121px">
<col style="width: 430px">
<col style="width: 390px">
<col style="width: 144px">
</colgroup>
<thead>
  <tr>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">参数名</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">输入/输出/属性</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">描述</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">数据类型</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">数据格式</span></th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">x1</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输入</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">矩阵乘运算中的左矩阵。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">x2</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">输入</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">矩阵乘运算中的右矩阵。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FRACTAL_NZ, ND</span></td>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">x1_scale_optional</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">可选输入</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">矩阵乘计算时，量化参数的缩放因子，对应公式的x1Scale。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT8_E8M0</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">x2_scale</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">输入</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">矩阵乘计算时，量化参数的缩放因子，对应公式的x2Scale。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT8_E8M0</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">bias_optional</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">可选输入</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">矩阵乘运算后累加的偏置，对应公式中的bias。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT32</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">y</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">输出</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">动态量化后的矩阵乘及激活计算结果。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">y_scale</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输出</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">动态量化后每个分组对应的量化尺度。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT8_E8M0</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
</tbody></table>

### 属性说明

| 属性名 | 描述 | 默认值 |
| --- | --- | --- |
| transpose_x1 | 表示x1的输入shape是否转置。 | false |
| transpose_x2 | 表示x2的输入shape是否转置。 | false |
| group_size | 用于输入m、n、k方向上的量化分组大小，由groupSizeM、groupSizeN、groupSizeK三个值拼接组成。当前MX场景仅支持[1, 1, 32]。 | 0 |
| activation_type | 激活函数类型，支持"gelu_tanh"、"gelu_erf"。 | "gelu_tanh" |
| y_dtype | 输出y的数据类型。 | DT_FLOAT8_E4M3FN |
| quant_mode | 量化模式，当前支持"mx"。 | "mx" |
| round_mode | 舍入模式。当y为FLOAT4_E2M1时支持"rint"、"floor"、"round"；当y为FLOAT8时仅支持"rint"。 | "rint" |
| scale_alg | 缩放算法。当y为FLOAT4_E2M1时支持0和2；当y为FLOAT8时支持0和1。 | 0 |
| dst_type_max | 目标数据类型最大值，用于量化范围控制。当scale_alg为0或1时不生效；当scale_alg为2时支持取值0.0和6.0-12.0。 | 0.0 |

## 约束说明

- 不支持空tensor。
- 支持连续tensor，[非连续tensor](../../docs/zh/context/non_contiguous_tensor.md)仅支持最后两根轴转置场景。
- 输入和输出支持以下数据类型组合:
    | x1            | x2            | x1_scale    | x2_scale    | bias         | y                         | y_scale      |
    |---------------|---------------|-------------|-------------|--------------|---------------------------|-------------|
    | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT8_E4M3FN             | FLOAT8_E8M0 |
    | FLOAT8_E5M2   | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT8_E5M2               | FLOAT8_E8M0 |
    | FLOAT8_E5M2   | FLOAT8_E5M2   | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT8_E5M2               | FLOAT8_E8M0 |
    | FLOAT8_E4M3FN | FLOAT8_E5M2   | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT8_E4M3FN             | FLOAT8_E8M0 |
    | FLOAT4_E2M1   | FLOAT4_E2M1   | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT4_E2M1               | FLOAT8_E8M0 |

  - x2为FLOAT8_E5M2时仅支持ND格式，不支持NZ格式。
  - y的数据类型由x1的数据类型决定，两者必须保持一致。
  - MXFP4场景约束（x1、x2、y数据类型均为FLOAT4_E2M1）：
    - scale_alg仅支持取值0和2。
    - 当scale_alg为2时，dst_type_max支持取值0.0和6.0-12.0。
    - round_mode支持"rint"、"floor"、"round"。
    - x1和x2的内轴必须为偶数，且k必须大于2。
    - y的N维度必须为偶数。
    - 当x2为NZ格式时，x1不支持转置。

## 调用说明

  | 调用方式   | 样例代码           | 说明                                         |
  | ---------------- | --------------------------- | --------------------------------------------------- |
  | aclnn接口（x2为NZ格式） | [test_aclnn_quant_matmul_activation_quant](examples/arch35/test_aclnn_quant_matmul_activation_quant.cpp) | 通过<br>[aclnnQuantMatmulActivationQuantWeightNz](docs/aclnnQuantMatmulActivationQuantWeightNz.md)<br>调用QuantMatmulActivationQuant算子。 |
  | aclnn接口（x2为ND格式） | [test_aclnn_quant_matmul_activation_quant_nd](examples/arch35/test_aclnn_quant_matmul_activation_quant_nd.cpp) | 通过<br>[aclnnQuantMatmulActivationQuant](docs/aclnnQuantMatmulActivationQuant.md)<br>调用QuantMatmulActivationQuant算子。 |
  | PyTorch API | - | 通过<br>[quant_matmul_activation_quant](docs/torchapi_quant_matmul_activation_quant.md)<br>调用QuantMatmulActivationQuant算子。 |
