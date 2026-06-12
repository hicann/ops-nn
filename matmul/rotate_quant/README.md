# RotateQuant

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                   |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |
| Atlas 200I/500 A2 推理产品                  |    ×     |
| Atlas 推理系列产品                          |    ×     |
| Atlas 训练系列产品                          |    ×     |

## 功能说明

- 算子功能：对张量x进行旋转变换，然后执行可选的clamp操作，最后执行对称动态量化（目的数据类型为int8或者quint4x2）或者MX量化（目的数据类型为FLOAT4类、FLOAT8类）。
- 计算公式：
  1. 旋转变换

    $$
    Y = (x.\text{reshape}(*,k) @ \text{rotation}).\text{reshape}(m, n)
    $$
    
    其中：$\mathbf{x} \in \mathbb{R}^{m \times n}$，$\mathbf{Y} \in \mathbb{R}^{m \times n}$，$\mathbf{rotation} \in \mathbb{R}^{k \times k}$。

  2. 当alpha在有效取值范围(0.0, 1.0)内，执行clamp计算

    $$
    groupMaxVal = GroupMax(|Y|) \\
    limit = alpha * groupMaxVal \\
    Y = Y.clamp(min=-limit, max=limit)
    $$
    GroupMax表示每32个为一组，计算组内最大值。
  
  3. 执行量化
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：对称动态量化（pertoken逐行量化）
      - 缩放因子计算（逐行计算）

        $$
        s_i = \frac{\max_{j \in [0,\ n-1]} |Y_{i,j}|}{C_{\text{MAX}}}
        $$

        其中：$s_i$ 是第 $i$ 行的缩放因子；$C_{\text{MAX}}$ 是量化范围最大值，int8取127，quint4x2取7。
      - 量化计算

        $$
        y_{i,j} = \frac{Y_{i,j}}{s_i}
        $$

    - <term>Ascend 950PR/Ascend 950DT</term>：MX量化

      - 场景1，当scaleAlg为0时：
        - 将输入x在axis维度上按k = 32个数分组，一组k个数  $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = 32

          $$
          shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
          mxscale = 2^{shared\_exp}\\
          P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space 32\\
          $$

        - ​量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。

        - emax: 对应数据类型的最大正则数的指数位。

            |   DataType    | emax |
            | :-----------: | :--: |
            |  FLOAT4_E2M1  |  2   |
            | FLOAT8_E4M3FN |  8   |
            |  FLOAT8_E5M2  |  15  |

      - 场景2，当scaleAlg为1时，只涉及FP8类型：
        - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
        - 找到该块中数值的最大绝对值:

          $$
          Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
          $$

        - 将FP32映射到目标数据类型FP8可表示的范围内，其中$Amax(DType)$是目标精度能表示的最大值

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
        - 当dstTypeMax = 6.0/7.0时：
          - 将输入x在axis维度上按k = 32个数分组，一组k个数  $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = 32：
          
            $$
            shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{如果} 尾数位的高比特前一/两位 \text{为1，且尾数不全为0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{其它} \end{cases} \\
            $$

            $$
            P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space 32\\
            $$

          - ​量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。
        - 当dstTypeMax != 6.0/7.0时：
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
          - ​量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。

            

## 参数说明
  
  <table style="undefined;table-layout: fixed; width: 962px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待旋转量化的输入张量。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rotation</td>
      <td>输入</td>
      <td>旋转矩阵。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
  <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>clamp需要限制的范围的缩放系数。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>量化后的输出张量。</td>
      <td>INT8、INT4、FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleOut</td>
      <td>输出</td>
      <td>动态量化计算出的缩放系数。</td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- <term>Ascend 950PR/Ascend 950DT</term>：
  - x的shape为(*, N)，维度范围[1, 7]；rotation的shape为(K, K)或(N/K, K, K)，维度范围[2, 3], K当前版本仅支持取32，64，128。
  - x最后一维的长度(N)必须是K的整数倍。
  - yOut的输出类型为FLOAT4_E2M1、FLOAT8_E4M3FN或FLOAT8_E5M2，shape与x相同。
  - x和rotation的数据类型必须相同。
  - scaleOut的shape必须是(*, CeilDiv(N,64), 2)，数据类型为FLOAT8_E8M0。
  - alpha为可选输入，不为空指针时，shape为(1,)，数据类型为BFLOAT16，有效取值范围(0.0, 1.0)。传入空指针或者不在有效取值范围内不做clamp处理。
  - axis目前只支持-1或者D-1，D为x的shape的维数。
  - roundMode支持"rint"、"round"、"floor"，传入空指针时，采用"rint"模式。当yOut的数据类型为FLOAT8_E4M3FN或FLOAT8_E5M2时，roundMode仅支持"rint"。
  - scaleAlg支持取值0、1、2，当yOut的数据类型为FLOAT8_E4M3FN或FLOAT8_E5M2时只支持0和1，当yOut的数据类型为FLOAT4_E2M1时只支持0和2。
  - dstTypeMax：当scaleAlg=2时dstTypeMax必须在[6.0, 12.0]范围内，其余场景仅支持0.0。
  - trans目前只支持false。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - x的shape为(M, N)，rotation的shape为(K, K)。
  - rotation的shape必须是方阵(K, K)。
  - x第二维的长度(N)必须是K的整数倍，N必须可以整除8。
  - 当yOut的输出类型为int8时，y的shape必须和x相同(M, N)。
  - 当yOut的输出类型为int32时，y的shape必须为(M, N//8)。
  - x和rotation的数据类型必须相同。
  - scaleOut的shape必须是(M)，数据类型为FLOAT。
  - alpha为可选输入，当前版本仅支持传入空指针。
  - axis目前只支持-1或者1。
  - roundMode支持"rint"、"round"、"floor"，传入空指针时，采用"rint"模式。
  - scaleAlg目前只支持0。
  - dstTypeMax目前只支持0.0。
  - trans目前只支持false。
  - N的范围为[128, 16000]。
  - K的范围为[16, 1024]。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_rotate_quant](examples/test_aclnn_rotate_quant.cpp) | 通过[aclnnRotateQuant](docs/aclnnRotateQuant.md)等方式调用RotateQuant算子。 |
