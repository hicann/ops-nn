# aclnnQuantMatmulActivationQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/matmul/quant_matmul_activation_quant)

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

- 接口功能：融合量化的矩阵乘、激活以及动态量化计算。当前支持激活为gelu（包括gelu_tanh和gelu_erf），量化模式为MX [量化模式](../../../docs/zh/context/quant_mode_introduction.md)。最小支持输入维度为2维，最大支持输入维度为6维。相似接口有aclnnMm（仅支持2维Tensor作为输入的矩阵乘）和aclnnBatchMatMul（仅支持三维的矩阵乘，其中第一维是Batch维度），本接口在其他接口的基础上融合了gelu激活及动态MX量化，提高硬件计算效率。

- 计算公式：

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：

    支持MX[量化模式](../../../docs/zh/context/quant_mode_introduction.md)，不同量化模式对应的输入输出数据类型组合参见[约束说明](#约束说明)。

    <details>

    <summary><strong>QuantMatmul MX量化模式</strong></summary>

      $$
      matmulOut[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice))* (x1Scale[m/gsM, j] * x2Scale[j, n/gsN]))+bias[n]
      $$

      其中，gsM，gsN和gsK分别代表groupSizeM，groupSizeN和groupSizeK；x1Slice代表x1第m行长度为groupSizeK的向量，x2Slice代表x2第n列长度为groupSizeK的向量；K轴均从j*groupSizeK起始切片，j的取值范围为[0, kLoops)，kLoops = ceil(K / groupSizeK)，K为K轴长度，支持最后的切片长度不足groupSizeK。仅MX量化模式下包含bias。

    </details>

    <details>

    <summary><strong>激活计算公式</strong></summary>

      - gelu_tanh(高性能近似)：

        $$
        activationOut=GELU(matmulOut)=matmulOut × Φ(matmulOut)=0.5 * matmulOut * (1 + tanh( \sqrt{2 / \pi} * (matmulOut + 0.044715 * matmulOut^{3})))
        $$

      - gelu_erf：

        $$
        activationOut=GELU(matmulOut)=0.5 * matmulOut * (1 + erf(matmulOut / \sqrt{2}))
        $$

    </details>

    <details>

    <summary><strong>动态量化计算公式</strong></summary>

      - 场景1，当scaleAlg为0时：
        - 将输入activationOut在尾轴上按$k=32$个数分组，一组k个数 $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\},\space k = 32$

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
          - 将输入x在axis维度上按k = blocksize个数分组，一组k个数  $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = blocksize：
          $$
          shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{如果} 尾数位的高比特前一/两位 \text{为1，且尾数不全为0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{其它} \end{cases} \\
          $$
          $$
          P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
          $$
          - ​量化后的$P_{i}$按对应的$V_{i}$的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。
        - 当dstTypeMax != 0.0/6.0/7.0时：
          - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
          - 找到该块中数值的最大绝对值:
          $$
          Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
          $$
          - 将FP32映射到目标数据类型可表示的范围内，其中当dst_max_value=0时，$Amax(DType)$是目标精度能表示的最大值；当dst_max_value!=0时，$Amax(DType)$是dst_max_value传入值。
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
          - ​量化后的$P_{i}$按对应的$V_{i}$的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。

    </details>

  <!-- end id7 -->

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnQuantMatmulActivationQuantGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnQuantMatmulActivationQuant"接口执行计算。

```cpp
aclnnStatus aclnnQuantMatmulActivationQuantGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *x1ScaleOptional,
  const aclTensor *x2Scale,
  const aclTensor *biasOptional,
  bool             transposeX1,
  bool             transposeX2,
  int64_t          groupSize,
  char            *activationType,
  char            *quantMode,
  char            *roundMode,
  int64_t          scaleAlg,
  double           dstTypeMax,
  aclTensor       *yOut,
  aclTensor       *yScaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulActivationQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantMatmulActivationQuantGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1554px"><colgroup>
  <col style="width: 248px">
  <col style="width: 121px">
  <col style="width: 210px">
  <col style="width: 327px">
  <col style="width: 250px">
  <col style="width: 115px">
  <col style="width: 138px">
  <col style="width: 145px">
  </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x1</td>
        <td>输入</td>
        <td>公式中的输入x1。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
            <li>支持最后两根轴转置情况下的<a href="../../../docs/zh/context/non_contiguous_tensor.md">非连续的Tensor</a>，其他轴方向不支持非连续的Tensor。</li>
            <li>transposeX1为false情况下x1各个维度表示：（batch, m, k），batch可不存在。</li>
            <li>transposeX1为true情况下x1各个维度表示：（batch, k, m），batch可不存在。</li>
          </ul>
        </td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
        <td>ND</td>
        <td>2-6</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>输入</td>
        <td>公式中的输入x2。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
            <li>支持最后两根轴转置情况下的<a href="../../../docs/zh/context/non_contiguous_tensor.md">非连续的Tensor</a>，其他轴方向不支持非连续的Tensor。</li>
            <li>transposeX2为false情况下x2各个维度表示：（batch, k, n），batch可不存在。</li>
            <li>transposeX2为true情况下x2各个维度表示：（batch, n, k），batch可不存在。</li>
          </ul>
        </td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
        <td>ND</td>
        <td>2-6</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>x1ScaleOptional</td>
        <td>可选输入</td>
        <td>公式中的输入x1Scale。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
            <li>当x1Scale为FLOAT8_E8M0时，x1Scale为3维，各个维度表示：transposeX1为false时为(m, ceil(k / 64), 2)，transposeX1为true时为(ceil(k / 64), m, 2)。</li>
          </ul>
        </td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1、3</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>x2Scale</td>
        <td>输入</td>
        <td>表示量化参数，公式中的输入x2Scale。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
            <li>当x2Scale为FLOAT8_E8M0时，x2Scale为3维，各个维度表示：transposeX2为false时为(ceil(k / 64), n, 2)，transposeX2为true时为(n, ceil(k / 64), 2)。</li>
          </ul>
        </td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1、3</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>可选输入</td>
        <td>公式中的输入bias。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
            <li>shape支持1维(n,)或3维(batch, 1, n)，n与x2的n一致。</li>
            <li>当y的shape为2、4、5、6维时，bias的shape只支持1维(n,)。</li>
            <li>如果不使用该参数，传入nullptr。</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1、3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>transposeX1</td>
        <td>输入</td>
        <td>表示x1的输入shape是否转置。</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX2</td>
        <td>输入</td>
        <td>表示x2的输入shape是否转置。</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize</td>
        <td>可选输入</td>
        <td>用于输入m、n、k方向上的量化分组大小。</td>
        <td>由3个方向的groupSizeM，groupSizeN，groupSizeK三个值拼接组成，每个值占16位，共占用int64_t类型groupSize的低48位（groupSize中的高16位的数值无效），计算公式见表格下方公式一。不支持groupSize的场景，传入0。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>activationType</td>
        <td>输入</td>
        <td>激活的类型。</td>
        <td>支持{"gelu_tanh", "gelu_erf"}。</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMode</td>
        <td>输入</td>
        <td>量化的模式。</td>
        <td>支持{"mx"}。</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>roundMode</td>
        <td>输入</td>
        <td>表示数据转换的模式，对应公式中的round_mode。</td>
        <td>
          <ul>
            <li>当yDtype为FLOAT4_E2M1时，支持{"rint", "floor", "round"}。</li>
            <li>当yDtype为FLOAT8_E4M3FN/FLOAT8_E5M2时，仅支持{"rint"}。</li>
          </ul>
        </td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleAlg</td>
        <td>输入</td>
        <td>表示mxscaleOut的计算方法，对应公式中的scaleAlg。</td>
        <td>
          <ul>
            <li>当yDtype为FLOAT4_E2M1时，支持取值0和2。取值为0代表场景1，为2代表场景3（使能dstTypeMax）。</li>
            <li>当yDtype为FLOAT8_E4M3FN/FLOAT8_E5M2时，支持取值0和1。取值为0代表场景1，为1代表场景2。</li>
          </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dstTypeMax</td>
        <td>输入</td>
        <td>表示maxType的取值，对应公式中的Amax(DType)。</td>
        <td>
          <ul>
            <li>当scaleAlg为0或1时，dstTypeMax不生效，传入0.0即可。</li>
            <li>当scaleAlg为2时，dstTypeMax支持取值0.0和6.0-12.0。取值为0.0表示使用目标类型的默认最大值，取值为6.0-12.0时使用指定值。</li>
          </ul>
        </td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
            <tr>
        <td>yOut</td>
        <td>输出</td>
        <td>公式中的输出y，动态量化后的矩阵乘及激活计算结果。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
            <li>shape和矩阵乘计算结果一致，(batch, m, n)，batch可不存在。</li>
          </ul>
        </td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
        <td>ND</td>
        <td>2-6</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yScaleOut</td>
        <td>输出</td>
        <td>动态量化后每个分组对应的量化尺度。</td>
        <td>
          <ul>
            <li>不支持空Tensor。</li>
            <li>shape在尾轴上为y对应轴的值除以32向上取整，并对其进行偶数pad，pad填充值为0。</li>
            <li>yScale输出需要对每两行数据进行交织处理。</li>
            <li>shape和矩阵乘计算结果一致，(batch, m, n/32, 2)，batch可不存在。</li>
          </ul>
        </td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1-8</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td style="white-space: nowrap">executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody></table>

  - 注：可选输入是指可选的量化参数，支持传入空指针。

  - 公式一：<a name='f1'></a>

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

    - x2支持最后两根轴转置情况下的[非连续的Tensor](../../../docs/zh/context/non_contiguous_tensor.md)，其他场景的[非连续的Tensor](../../../docs/zh/context/non_contiguous_tensor.md)不支持。
    - 支持groupSize传非0。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
  </colgroup>
  <thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr></thead>
      <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>传入的x1、x2、x1Scale、x2Scale、yOut或yScaleOut是空指针。</td>
      </tr>
      <tr>
        <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="4">161002</td>
        <td>x1、x2、biasOptional、x1ScaleOptional、x2Scale、yOut或yScaleOut的数据类型和数据格式不在支持的范围之内。</td>
      </tr>
      <tr>
        <td>x1、x2、biasOptional、x1ScaleOptional、x2Scale、yOut或yScaleOut的shape不满足校验条件。</td>
      </tr>
      <tr>
        <td>x1、x2、biasOptional、x1ScaleOptional、x2Scale、yOut或yScaleOut是空tensor。</td>
      </tr>
      <tr>
        <td>activationType、quantMode、roundMode、scaleAlg、dstTypeMax、groupSize、transposeX1的值不在支持范围之内。</td>
      </tr>
    </tbody></table>

## aclnnQuantMatmulActivationQuant

- **参数说明**

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    </tr></thead>
  <tbody>
  <tr>
    <td>workspace</td>
    <td>输入</td>
    <td>在Device侧申请的workspace内存地址。</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>输入</td>
    <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantMatmulActivationQuantGetWorkspaceSize获取。</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>输入</td>
    <td>op执行器，包含了算子计算流程。</td>
  </tr>
  <tr>
    <td>stream</td>
    <td>输入</td>
    <td>指定执行任务的Stream。</td>
  </tr>
    </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnQuantMatmulActivationQuant默认确定性实现。

<details>

- **公共约束：**
  <a id="公共约束"></a>
  - x1仅支持ND格式，x2仅支持ND格式。
  - transposeX1为false时x1的shape：(batch, m, k)。transposeX1为true时x1的shape：(batch, k, m)。其中batch代表前0~4维，0维表示batch不存在。
  - transposeX2为false时x2的shape：(batch, k, n)。transposeX2为true时x2的shape：(batch, n, k)。其中batch代表前0~4维，0维表示batch不存在。k与x1的shape中的k一致。
  - x1支持最后两根轴转置情况下的[非连续的Tensor](../../../docs/zh/context/non_contiguous_tensor.md)，其他场景的[非连续的Tensor](../../../docs/zh/context/non_contiguous_tensor.md)不支持。
  - x2支持最后两根轴转置情况下的[非连续的Tensor](../../../docs/zh/context/non_contiguous_tensor.md)，其他场景的[非连续的Tensor](../../../docs/zh/context/non_contiguous_tensor.md)不支持。
  - bias相关约束：
    - 可选参数，支持传入空指针, 不支持空tensor。
    - 数据类型必须为FLOAT32。
    - shape支持1维(n,)或3维(batch, 1, n)，n与x2的n一致。
    - 当y的shape为2、4、5、6维时，bias的shape只支持1维(n,)。
  - groupSize相关约束：
    - 仅在MX[量化模式](../../../docs/zh/context/quant_mode_introduction.md)中生效。
    - 传入的groupSize内部会按如下公式分解得到groupSizeM、groupSizeN、groupSizeK, 当前支持[0, 0, 0]或[1, 1, 32]的组合, [0, 0, 0]会自动推导为[1, 1, 32]。

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

  - y的shape支持2~6维，(batch, m, n)，batch可不存在，m与x1的m一致，n与x2的n一致。

  <details>

  <summary><strong>MX量化场景约束：</strong></summary>
  <a id="MX量化"></a>

  - 输入和输出支持以下数据类型组合：
  <a id="输入和输出支持以下数据类型组合MX"></a>

      | x1            | x2            | x1Scale     | x2Scale     | bias         | y                         | yScale      |
      |---------------|---------------|-------------|-------------|--------------|---------------------------|-------------|
      | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT8_E4M3FN             | FLOAT8_E8M0 |
      | FLOAT8_E5M2   | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT8_E5M2               | FLOAT8_E8M0 |
      | FLOAT8_E5M2   | FLOAT8_E5M2   | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT8_E5M2               | FLOAT8_E8M0 |
      | FLOAT8_E4M3FN | FLOAT8_E5M2   | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT8_E4M3FN             | FLOAT8_E8M0 |
      | FLOAT4_E2M1   | FLOAT4_E2M1   | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT4_E2M1               | FLOAT8_E8M0 |

  - x1数据类型、x2数据类型、x1、x2、x1Scale、x2Scale和groupSize的取值关系：

      |量化类型|x1数据类型|x2数据类型|x1 shape|x2 shape|x1Scale shape|x2Scale shape|bias shape|yScale shape|[gsM, gsN, gsK]|groupSize|
      |-------|--------|--------|--------|--------|-------------|-------------|------------|---------------------------------------|--|--|
      |MX全量化|FLOAT8_E4M3FN|FLOAT8_E4M3FN|<li>非转置：(batch_x1, m, k)</li>|<li>非转置：(batch_x2, k, n)</li><li>转置：(batch_x2, n, k)</li>|<li>非转置：(batch_x1, m, ceil(k / 64), 2)</li>|<li>非转置：(batch_x2, ceil(k / 64), n, 2)</li><li>转置：(batch_x2, n, ceil(k / 64), 2)</li>|(n,)或(batch_max, 1, n)|null|[1, 1, 32]|4295032864|
      |MX全量化|FLOAT8_E5M2|FLOAT8_E4M3FN|<li>非转置：(batch_x1, m, k)</li>|<li>非转置：(batch_x2, k, n)</li><li>转置：(batch_x2, n, k)</li>|<li>非转置：(batch_x1, m, ceil(k / 64), 2)</li>|<li>非转置：(batch_x2, ceil(k / 64), n, 2)</li><li>转置：(batch_x2, n, ceil(k / 64), 2)</li>|(n,)或(batch_max, 1, n)|null|[1, 1, 32]|4295032864|
      |MX全量化|FLOAT8_E5M2|FLOAT8_E5M2|<li>非转置：(batch_x1, m, k)</li>|<li>非转置：(batch_x2, k, n)</li><li>转置：(batch_x2, n, k)</li>|<li>非转置：(batch_x1, m, ceil(k / 64), 2)</li>|<li>非转置：(batch_x2, ceil(k / 64), n, 2)</li><li>转置：(batch_x2, n, ceil(k / 64), 2)</li>|(n,)或(batch_max, 1, n)|null|[1, 1, 32]|4295032864|
      |MX全量化|FLOAT8_E4M3FN|FLOAT8_E5M2|<li>非转置：(batch_x1, m, k)</li>|<li>非转置：(batch_x2, k, n)</li><li>转置：(batch_x2, n, k)</li>|<li>非转置：(batch_x1, m, ceil(k / 64), 2)</li>|<li>非转置：(batch_x2, ceil(k / 64), n, 2)</li><li>转置：(batch_x2, n, ceil(k / 64), 2)</li>|(n,)或(batch_max, 1, n)|null|[1, 1, 32]|4295032864|
      |MX全量化|FLOAT4_E2M1|FLOAT4_E2M1|<li>非转置：(batch_x1, m, k)</li><li>转置：(batch_x1, k, m)</li>|<li>非转置：(batch_x2, k, n)</li><li>转置：(batch_x2, n, k)</li>|<li>非转置：(batch_x1, m, ceil(k / 64), 2)</li><li>转置：(batch_x1, ceil(k / 64), m, 2)</li>|<li>非转置：(batch_x2, ceil(k / 64), n, 2)</li><li>转置：(batch_x2, n, ceil(k / 64), 2)</li>|(n,)或(batch_max, 1, n)|null|[1, 1, 32]|4295032864|

  - 注：上表中gsM、gsK和gsN分别表示groupSizeM、groupSizeK和groupSizeN。
  - MX量化场景下，x1和x1Scale的转置属性需要保持一致，x2和x2Scale的转置属性需要保持一致。
  - yOut的数据类型由x1的数据类型决定，两者必须保持一致。
  - MXFP4场景约束（x1、x2、y数据类型均为FLOAT4_E2M1）：
    - scaleAlg仅支持取值0和2。
    - 当scaleAlg为2时，dstTypeMax支持取值0.0和6.0-12.0。
    - roundMode支持{"rint", "floor", "round"}。
    - x1和x2的内轴必须为偶数，且k必须大于2。
    - y的N维度必须为偶数。

  </details>

</details>

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

x1为FLOAT8_E4M3FN，x2为FLOAT8_E4M3FN（ND格式），x1Scale为FLOAT8_E8M0，x2Scale为FLOAT8_E8M0，激活为gelu_tanh，scaleAlg为0（场景1）。

  ```cpp
  #include <iostream>
  #include <memory>
  #include <cmath>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_activation_quant.h"
  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)
  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)
  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)
  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }
  int Init(int32_t deviceId, aclrtStream* stream)
  {
      // 固定写法，资源初始化
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }
  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }
      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }
  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }
  // 将float8_e4m3的uint8_t表示转换为float表示
  float Fp8E4M3ToFloat(uint8_t h)
  {
      int sign = (h >> 7) & 0x1;
      int exponent = (h >> 3) & 0xF;
      int mantissa = h & 0x7U;
      float value = 0.0f;
      if (exponent == 0) {
          if (mantissa == 0) {
              return sign ? -0.0f : 0.0f;
          } else {
              value = static_cast<float>(mantissa) / 8.0f;
              value = ldexp(value, -6);
          }
      } else {
          value = static_cast<float>(mantissa) / 8.0f + 1.0f;
          value = ldexp(value, exponent - 7);
      }
      return sign ? -value : value;
  }

  float Fp8E8M0ToFloat(uint8_t h)
  {
      uint32_t exponent = h & 0x00FFU;
      uint32_t fBits = exponent << 23;
      return *reinterpret_cast<float*>(&fBits);
  }
  int AclnnQuantMatmulActivationQuantTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
      // 2. 构造输入与输出，需要根据API的接口自定义构造
      int64_t m = 5;
      int64_t k = 64;
      int64_t n = 128;
      bool transposeX1 = false;
      bool transposeX2 = false;
      int64_t groupSize = 32;
      std::vector<int64_t> x1Shape = {m, k};
      std::vector<int64_t> x2Shape = {k, n};
      std::vector<int64_t> x1ScaleShape = {m, k / groupSize / 2, 2};
      std::vector<int64_t> x2ScaleShape = {k / groupSize / 2, n, 2};
      std::vector<int64_t> outShape = {m, n};
      std::vector<int64_t> outScaleShape = {m, n / groupSize / 2, 2};
      void* x1DeviceAddr = nullptr;
      void* x2DeviceAddr = nullptr;
      void* x1ScaleDeviceAddr = nullptr;
      void* x2ScaleDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      void* outScaleDeviceAddr = nullptr;
      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* x1Scale = nullptr;
      aclTensor* x2Scale = nullptr;
      aclTensor* bias = nullptr;
      aclTensor* out = nullptr;
      aclTensor* outScale = nullptr;
      std::vector<uint8_t> x1HostData(m * k, 0b00111000);                  // float8_e4m3的1.0
      std::vector<uint8_t> x2HostData(k * n, 0b00111000);                  // float8_e4m3的1.0
      std::vector<uint8_t> x1ScaleHostData(m * k / groupSize, 0b01111111); // float8_e8m0的1.0
      std::vector<uint8_t> x2ScaleHostData(k * n / groupSize, 0b01111111); // float8_e8m0的1.0
      std::vector<uint8_t> outHostData(m * n, 0);
      std::vector<uint8_t> outScaleHostData(m * n / groupSize, 0);
      // 创建x1 aclTensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2 aclTensor
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x1Scale aclTensor
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2Scale aclTensor
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建mxscale aclTensor
      ret = CreateAclTensor(outScaleHostData, outScaleShape, &outScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &outScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outScaleTensorPtr(outScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> outScaleDeviceAddrPtr(outScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 3. 调用CANN算子库API，需要修改为具体的API名称
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
    int64_t groupSizeValue = 4295032864;
      ret = aclnnQuantMatmulActivationQuantGetWorkspaceSize(x1, x2, x1Scale, x2Scale, nullptr,
                  transposeX1, transposeX2, groupSizeValue, "gelu_tanh", "mx", "rint", 0, 0.0, out, outScale, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulActivationQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnQuantMatmulActivationQuant第二段接口
      ret = aclnnQuantMatmulActivationQuant(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulActivationQuant failed. ERROR: %d\n", ret); return ret);
      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<uint8_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %f\n", i, Fp8E4M3ToFloat(resultData[i]));
      }
      size = GetShapeSize(outScaleShape);
      std::vector<uint8_t> scaleData(size, 0);
      ret = aclrtMemcpy(scaleData.data(), scaleData.size() * sizeof(scaleData[0]), outScaleDeviceAddr,
                        size * sizeof(scaleData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("scale[%ld] is: %f\n", i, Fp8E8M0ToFloat(scaleData[i]));
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = AclnnQuantMatmulActivationQuantTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnQuantMatmulActivationQuantTest failed. ERROR: %d\n", ret); return ret);
      Finalize(deviceId, stream);
      return 0;
  }
  ```
